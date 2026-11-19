"""Point encoder and goal-auxiliary TD3 updates."""
import math
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from ga_env import ACT_DIM, GOAL_SCALE, POINTS, POS_SCALE, ROT_SCALE
from pointnet2 import PointEncoder


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = PointEncoder()
        self.trunk = nn.Sequential(nn.Linear(256, 256), nn.ReLU(),
                                   nn.Linear(256, 256), nn.ReLU())
        self.action = nn.Linear(256, ACT_DIM)
        self.goal = nn.Linear(256, 6)

    def forward(self, obs):
        features = self.trunk(self.encoder(obs))
        return self.action(features).tanh(), self.goal(features)


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = PointEncoder()
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(256 + ACT_DIM, 256), nn.ReLU(),
                          nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))
            for _ in range(2)])
        self.goal = nn.Sequential(nn.Linear(256, 256), nn.ReLU(),
                                  nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 6))

    def forward(self, obs, action):
        features = self.encoder(obs)
        joined = torch.cat((features, action), dim=1)
        q1, q2 = (head(joined).squeeze(1) for head in self.heads)
        return q1, q2, self.goal(features)


def pose_error(predicted, target, translation_scale, rotation_scale, keypoints):
    """Mean absolute error on virtual gripper points after rigid transforms."""
    predicted, target = predicted.float(), target.float()

    def transform(pose):
        axis_angle = pose[:, None, 3:6] * rotation_scale
        angle = axis_angle.norm(dim=-1, keepdim=True)
        points = keypoints[None].expand(pose.shape[0], -1, -1)
        axis_angle = axis_angle.expand_as(points)
        cross = torch.linalg.cross(axis_angle, points)
        rotated = (points + torch.sinc(angle / math.pi) * cross +
                   0.5 * torch.sinc(angle / (2 * math.pi)).square() *
                   torch.linalg.cross(axis_angle, cross))
        return rotated + pose[:, None, :3] * translation_scale

    return (transform(predicted) - transform(target)).abs().mean(dim=(1, 2)) / translation_scale


def masked_mean(value, mask):
    return (value * mask).sum() / mask.sum().clamp_min(1)


def scheduled_learning_rate(step, start, peak, ramp_end, decay_start, end):
    if step < ramp_end:
        return start + (peak - start) * (step / ramp_end)
    if step <= decay_start:
        return peak
    progress = min(1., (step - decay_start) / (end - decay_start))
    return peak * (1. + math.cos(math.pi * progress)) / 2.


class Learner:
    def __init__(self, device, lr, auxiliary_weight, *, cuda_graph=False, lr_warmup_updates=0,
                 lr_schedule=None, env_steps_per_update=None, gamma=0.95, tau=1e-4, ddpg_weight=0.2,
                 lr_milestones=None, lr_gamma=0.5):
        self.gamma = gamma
        self.tau = tau
        self.ddpg_weight = ddpg_weight
        self.actor = Actor().to(device)
        self.critic = Critic().to(device)
        self.actor_target = deepcopy(self.actor).eval().requires_grad_(False)
        self.critic_target = deepcopy(self.critic).eval().requires_grad_(False)
        optimizer_options = dict(lr=lr, fused=device.type == 'cuda', capturable=cuda_graph)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), **optimizer_options)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), **optimizer_options)
        self.lr_schedule = lr_schedule
        self.lr_milestones = tuple(sorted(lr_milestones)) if lr_milestones else None
        self.lr_gamma = lr_gamma
        self.env_steps_per_update = env_steps_per_update
        self.learning_rate = lr
        if (lr_schedule or self.lr_milestones) and cuda_graph:
            init_lr = lr_schedule[0] if lr_schedule else lr
            for optimizer in (self.actor_opt, self.critic_opt):
                optimizer.param_groups[0]['lr'] = torch.tensor(init_lr, device=device)
        self.base_lr = lr
        self.lr_warmup_updates = lr_warmup_updates
        self.keypoints = torch.tensor([
            [0., 0., 0.], [0., 0.04, 0.], [0., -0.04, 0.],
            [0., 0.04, 0.05], [0., -0.04, 0.05],
        ], device=device, dtype=torch.float32)
        self.auxiliary_weight = auxiliary_weight
        self.updates = 0
        self.graph_replays = 0
        self._warmup_stream = torch.cuda.Stream(device=device) if cuda_graph else None
        self._warmup_updates = 0
        self._graphs = {}
        self._graph_inputs = None

    def imitation_loss(self, pred, action):
        return (pose_error(pred, action, POS_SCALE, ROT_SCALE, self.keypoints)
                + F.smooth_l1_loss(pred[:, 6], action[:, 6], reduction='none'))

    def pretrain(self, batch):
        obs, actions, label, _, _, _, _, _ = batch
        predicted, _ = self.actor(obs)
        loss = self.imitation_loss(predicted, label).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.actor_opt.step()
        return loss.detach()

    def update(self, batch, *, target_noise=None):
        if self.lr_milestones:
            next_update = self.updates + 1
            exponents = sum(next_update >= m for m in self.lr_milestones)
            self.learning_rate = self.base_lr * (self.lr_gamma ** exponents)
            for optimizer in (self.actor_opt, self.critic_opt):
                group = optimizer.param_groups[0]
                if isinstance(group['lr'], torch.Tensor):
                    if group['lr'].device != batch[0].device:
                        group['lr'] = group['lr'].to(batch[0].device)
                    group['lr'].fill_(self.learning_rate)
                else:
                    group['lr'] = self.learning_rate
        elif self.lr_schedule:
            next_step = (self.updates + 1) * self.env_steps_per_update if self.updates else 0
            self.learning_rate = scheduled_learning_rate(next_step, *self.lr_schedule)
            for optimizer in (self.actor_opt, self.critic_opt):
                group = optimizer.param_groups[0]
                if isinstance(group['lr'], torch.Tensor):
                    if group['lr'].device != batch[0].device:
                        group['lr'] = group['lr'].to(batch[0].device)
                    group['lr'].fill_(self.learning_rate)
                else:
                    group['lr'] = self.learning_rate
        update_actor = (self.updates + 1) % 2 == 0
        noise = torch.randn_like(batch[1]) if target_noise is None else target_noise
        if self.updates < self.lr_warmup_updates:
            # Adam's scalar learning rate is fixed into a captured graph.
            fraction = self.updates / max(1, self.lr_warmup_updates - 1)
            lr = self.base_lr * (0.1 + 0.9 * fraction)
            self.learning_rate = lr
            for optimizer in (self.actor_opt, self.critic_opt):
                for group in optimizer.param_groups:
                    group['lr'] = lr
            metrics = self._update_batch(batch, noise, update_actor)
        elif self._warmup_stream is None:
            metrics = self._update_batch(batch, noise, update_actor)
        elif self._warmup_updates < 4:
            # Initialize optimizer state for both paths with real, counted batches.
            current = torch.cuda.current_stream(device=noise.device)
            self._warmup_stream.wait_stream(current)
            with torch.cuda.stream(self._warmup_stream):
                metrics = self._update_batch(batch, noise, update_actor)
            current.wait_stream(self._warmup_stream)
            self._warmup_updates += 1
        else:
            metrics = self._replay_update(batch, noise, update_actor)
        self.updates += 1
        return metrics

    def _replay_update(self, batch, noise, update_actor):
        inputs = (*batch, noise)
        if self._graph_inputs is None:
            self._graph_inputs = tuple(torch.empty_like(value) for value in inputs)
        for static, value in zip(self._graph_inputs, inputs, strict=True):
            if static.shape != value.shape or static.dtype != value.dtype or static.device != value.device:
                raise ValueError('CUDA Graph requires fixed batch tensor shapes, dtypes and devices')
            static.copy_(value, non_blocking=True)
        if update_actor not in self._graphs:
            torch.cuda.current_stream(device=noise.device).synchronize()
            graph = torch.cuda.CUDAGraph()
            # Keep separate graph pools; either path can replay next.
            with torch.cuda.graph(graph):
                metrics = self._update_batch(self._graph_inputs[:-1], self._graph_inputs[-1], update_actor)
            self._graphs[update_actor] = graph, metrics
        graph, metrics = self._graphs[update_actor]
        graph.replay()
        self.graph_replays += 1
        return metrics

    def _update_batch(self, batch, target_noise, update_actor):
        obs, action, label, reward, next_obs, terminal, goal, masks = batch
        is_demo, has_goal = masks.unbind(dim=1)
        with torch.no_grad():
            next_action, _ = self.actor_target(next_obs)
            noise = (target_noise * 0.15).clamp(-0.3, 0.3)
            q1, q2, _ = self.critic_target(next_obs, (next_action + noise).clamp(-1, 1))
            target = (reward + self.gamma * (1 - terminal) * torch.minimum(q1, q2)).clamp(0., 1.)
        q1, q2, predicted_goal = self.critic(obs, action)
        td = F.smooth_l1_loss(q1, target) + F.smooth_l1_loss(q2, target)
        critic_aux = td.new_zeros(())
        if self.auxiliary_weight:
            critic_aux = masked_mean(
                pose_error(predicted_goal, goal, GOAL_SCALE, math.pi, self.keypoints), has_goal)
        self.critic_opt.zero_grad(set_to_none=True)
        (td + self.auxiliary_weight * critic_aux).backward()
        self.critic_opt.step()
        metrics = {'td_loss': td.detach(), 'critic_aux_loss': critic_aux.detach()}
        if update_actor:
            self.critic.requires_grad_(False)
            self.critic.eval()
            predicted, predicted_goal = self.actor(obs)
            q1_pi, q2_pi, _ = self.critic(obs, predicted)
            rl_loss = -masked_mean(torch.minimum(q1_pi, q2_pi), 1. - is_demo)
            bc = masked_mean(self.imitation_loss(predicted, label), is_demo)
            actor_aux = bc.new_zeros(())
            if self.auxiliary_weight:
                actor_aux = masked_mean(
                    pose_error(predicted_goal, goal, GOAL_SCALE, math.pi, self.keypoints), has_goal)
            loss = (self.ddpg_weight * rl_loss + (1. - self.ddpg_weight) * bc
                    + self.auxiliary_weight * actor_aux)
            self.actor_opt.zero_grad(set_to_none=True)
            loss.backward()
            self.actor_opt.step()
            self.critic.requires_grad_(True)
            self.critic.train()
            with torch.no_grad():
                for target_net, net in ((self.critic_target, self.critic), (self.actor_target, self.actor)):
                    torch._foreach_lerp_(list(target_net.parameters()), list(net.parameters()), self.tau)
                    # BN statistics are estimates, not trainable Polyak parameters.
                    torch._foreach_copy_(list(target_net.buffers()), list(net.buffers()))
            metrics.update(bc_loss=bc.detach(), actor_aux_loss=actor_aux.detach())
        return metrics
