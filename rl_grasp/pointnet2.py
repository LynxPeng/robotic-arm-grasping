"""Half-width PointNet++ using upstream CUDA set abstraction and BatchNorm.

Falls back to a state-dict-compatible pure-PyTorch PointnetSAModule when the
native pointnet2_ops CUDA extension is unavailable (e.g. Windows demos/eval).
Set GA_SMOKE_ENCODER=1 to substitute the lightweight smoke encoder instead.
"""
import os

import torch
from torch import nn


def _furthest_point_sample(xyz, npoint):
    """Deterministic furthest-point sampling matching pointnet2_ops semantics."""
    bsz, num_pts, _ = xyz.shape
    device = xyz.device
    centroids = torch.zeros(bsz, npoint, dtype=torch.long, device=device)
    distance = torch.full((bsz, num_pts), 1e10, dtype=xyz.dtype, device=device)
    farthest = torch.zeros(bsz, dtype=torch.long, device=device)
    batch_indices = torch.arange(bsz, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].unsqueeze(1)
        dist = ((xyz - centroid) ** 2).sum(dim=-1)
        distance = torch.minimum(distance, dist)
        farthest = distance.argmax(dim=-1)
    return centroids


def _ball_query(radius, nsample, xyz, new_xyz):
    """Pure-PyTorch ball query returning (B, npoint, nsample) point indices."""
    bsz, num_pts, _ = xyz.shape
    npoint = new_xyz.shape[1]
    sq_dists = ((new_xyz.unsqueeze(2) - xyz.unsqueeze(1)) ** 2).sum(dim=-1)
    idx = torch.arange(num_pts, device=xyz.device).view(1, 1, num_pts).expand(bsz, npoint, num_pts).clone()
    out_of_ball = sq_dists > radius * radius
    idx[out_of_ball] = num_pts
    take = min(nsample, num_pts)
    group_idx = idx.sort(dim=-1).values[:, :, :take]
    first_idx = group_idx[:, :, :1].expand(bsz, npoint, take)
    first_idx = torch.where(first_idx == num_pts, torch.zeros_like(first_idx), first_idx)
    group_idx = torch.where(group_idx == num_pts, first_idx, group_idx)
    if take < nsample:
        pad = group_idx[:, :, :1].expand(bsz, npoint, nsample - take)
        group_idx = torch.cat((group_idx, pad), dim=-1)
    return group_idx


def _gather_points(points, idx):
    """Gather (B, N, C) points at (B, M) or (B, M, K) indices."""
    bsz = points.shape[0]
    channels = points.shape[2]
    flat_idx = idx.reshape(bsz, -1, 1).expand(-1, -1, channels)
    gathered = torch.gather(points, 1, flat_idx)
    return gathered.reshape(bsz, *idx.shape[1:], channels)


class _PurePointnetSAModule(nn.Module):
    """Pure-PyTorch PointnetSAModule with identical parameter names to pointnet2_ops."""

    def __init__(self, mlp, npoint=None, radius=None, nsample=None, bn=True):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        layers = []
        in_channels = mlp[0] + 3  # Concatenated relative xyz coordinates.
        for out_channels in mlp[1:]:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=not bn))
            if bn:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        self.mlps = nn.ModuleList([nn.Sequential(*layers)])

    def forward(self, xyz, features):
        if self.npoint is None:
            new_xyz = None
            grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)
            grouped_features = features.unsqueeze(2)
        else:
            fps_idx = _furthest_point_sample(xyz, self.npoint)
            new_xyz = _gather_points(xyz, fps_idx)
            group_idx = _ball_query(self.radius, self.nsample, xyz, new_xyz)
            grouped_xyz = _gather_points(xyz, group_idx).permute(0, 3, 1, 2) - new_xyz.transpose(1, 2).unsqueeze(-1)
            grouped_features = _gather_points(features.transpose(1, 2), group_idx).permute(0, 3, 1, 2)
        new_features = torch.cat((grouped_xyz, grouped_features), dim=1)
        new_features = self.mlps[0](new_features).max(dim=-1).values
        return new_xyz, new_features


class PointEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        if os.environ.get('GA_SMOKE_ENCODER'):
            # Smoke-only substitute: per-point MLP + max-pool + BN projection.
            self._smoke = True
            self.point_mlp = nn.Sequential(nn.Linear(3, 64), nn.ReLU(),
                                           nn.Linear(64, 128), nn.ReLU(),
                                           nn.Linear(128, 256))
            self.features = nn.Sequential(nn.Linear(258, 256), nn.BatchNorm1d(256), nn.ReLU())
            return
        try:
            from pointnet2_ops.pointnet2_modules import PointnetSAModule
            self._native = True
        except ImportError:
            PointnetSAModule = _PurePointnetSAModule
            self._native = False
        self._smoke = False
        # Original GA-DDPG radii and sampling sizes; half-width MLPs.
        self.sa1 = PointnetSAModule(npoint=32, radius=.02, nsample=64,
                                    mlp=[3, 32, 32, 64], bn=True)
        self.sa2 = PointnetSAModule(npoint=32, radius=.04, nsample=128,
                                    mlp=[64, 64, 64, 128], bn=True)
        self.sa3 = PointnetSAModule(mlp=[128, 128, 128, 256], bn=True)
        # BatchNorm cancels preceding biases; omit redundant, noisy gradients.
        self.features = nn.Sequential(nn.Linear(258, 512, bias=False), nn.BatchNorm1d(512), nn.ReLU(),
                                      nn.Linear(512, 256, bias=False), nn.BatchNorm1d(256), nn.ReLU())

    def forward(self, obs):
        from ga_env import POINT_SCALE
        if self._native and not obs.is_cuda:
            raise ValueError('Native PointNet++ requires CUDA observations')
        points = obs[:, :-2].reshape(obs.shape[0], -1, 3)
        if self._smoke:
            xyz = points * POINT_SCALE
            feature = self.point_mlp(xyz).max(dim=1).values
            return self.features(torch.cat((feature, obs[:, -2:]), dim=-1))
        xyz = (points * POINT_SCALE).contiguous()
        features = points.transpose(1, 2).contiguous()
        xyz, features = self.sa1(xyz, features)
        xyz, features = self.sa2(xyz, features)
        _, features = self.sa3(xyz, features)
        return self.features(torch.cat((features.squeeze(-1), obs[:, -2:]), dim=-1))
