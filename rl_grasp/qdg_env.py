"""Mesh scenes shared by QDGset demonstration, training and evaluation."""
from dataclasses import dataclass
from hashlib import file_digest, sha256
import json
import math
from pathlib import Path

import numpy as np
import pybullet as p

from ga_env import PointGraspEnv, matrix


@dataclass
class MeshAsset:
    key: str
    mesh: Path
    bounds: np.ndarray
    inertia_origin: np.ndarray
    mass: float
    grasps: np.ndarray | None = None


def load_manifest(folder, split):
    folder = Path(folder).resolve()
    rows = json.loads((folder / f'{split}.json').read_text(encoding='utf-8'))
    grasps_path = folder / 'grasps.npz'
    grasps_archive = np.load(grasps_path, allow_pickle=False) if grasps_path.is_file() else None
    assets = {}
    try:
        for row in rows:
            key = row['object']
            mesh = (folder / row['mesh']).resolve()
            bounds = np.asarray(row['bounds'], dtype=float)
            inertia = np.asarray(row['inertia_origin'], dtype=float)
            mass = float(row['mass'])
            if (key in assets or not mesh.is_relative_to(folder) or not mesh.is_file()
                    or bounds.shape != (2, 3) or not np.isfinite(bounds).all()
                    or not np.all(bounds[1] > bounds[0]) or inertia.shape != (7,)
                    or not np.isfinite(inertia).all() or abs(np.linalg.norm(inertia[3:]) - 1) > .01
                    or not math.isfinite(mass) or mass <= 0):
                raise ValueError(f'Invalid QDGset asset {key!r} in {split}.json')
            grasps = (np.asarray(grasps_archive[key], dtype=float)
                      if grasps_archive is not None and key in grasps_archive else None)
            assets[key] = MeshAsset(key, mesh, bounds, inertia, mass, grasps=grasps)
    finally:
        if grasps_archive is not None:
            grasps_archive.close()
    if not assets:
        raise ValueError(f'Empty QDGset {split} split')
    return assets


def dataset_signature(folder):
    train, test = (load_manifest(folder, split) for split in ('train', 'holdout'))
    if train.keys() & test.keys():
        raise ValueError('A QDGset object ID appears in both splits')
    result = sha256()
    meshes = set()
    for split, assets in (('train', train), ('holdout', test)):
        result.update((Path(folder) / f'{split}.json').read_bytes())
        for asset in assets.values():
            with asset.mesh.open('rb') as stream:
                digest = file_digest(stream, 'sha256').digest()
            if digest in meshes:
                raise ValueError(f'QDGset object {asset.key!r} references the same mesh as another object')
            meshes.add(digest)
            result.update(digest)
    return result.hexdigest()


class QDGPointGraspEnv(PointGraspEnv):
    def __init__(self, asset=None, *, assets=None, **kwargs):
        if (asset is None) == (assets is None):
            raise ValueError('Provide a single demonstration asset or a split of online assets')
        self.fixed_asset = asset
        self.assets = assets
        self.asset_keys = tuple(assets or ())
        try:
            super().__init__(**kwargs)
        except Exception:
            if hasattr(self, 'client'):
                self.close()
            raise

    def _spawn_object(self, x, y):
        self.kind = self.asset.key
        yaw = p.getQuaternionFromEuler((0, 0, self.rng.uniform(-math.pi, math.pi)))
        xy = np.array((x, y, 0.)) - matrix(yaw) @ np.r_[self.mesh_center[:2], 0.]
        xy[2] = -self.asset.bounds[0, 2] + .005
        # A dynamic GEOM_MESH uses a convex hull; the RGB-D camera sees the authored mesh.
        collision = p.createCollisionShape(p.GEOM_MESH, fileName=str(self.asset.mesh))
        visual = p.createVisualShape(p.GEOM_MESH, fileName=str(self.asset.mesh),
                                     rgbaColor=(.94, .57, .10, 1))
        return p.createMultiBody(
            baseMass=self.asset.mass, baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual, basePosition=xy, baseOrientation=yaw,
            baseInertialFramePosition=self.asset.inertia_origin[:3],
            baseInertialFrameOrientation=self.asset.inertia_origin[3:])

    def _obj_pos(self):
        origin, quat = p.multiplyTransforms(
            *p.getBasePositionAndOrientation(self.obj), *self.inertia_to_mesh)
        return np.asarray(origin) + matrix(quat) @ self.mesh_center

    def reset(self, seed=None, kind=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if self.fixed_asset is not None:
            self.asset = self.fixed_asset
        else:
            key = kind if kind is not None else self.asset_keys[self.rng.integers(len(self.asset_keys))]
            self.asset = self.assets[key]
        self.mesh_center = self.asset.bounds.mean(axis=0)
        inertia = self.asset.inertia_origin
        self.inertia_to_mesh = p.invertTransform(inertia[:3], inertia[3:])
        # Keep the chosen object on a retry so difficult reset geometry cannot change the split.
        for attempt in range(1 if self.fixed_asset is not None else 5):
            try:
                obs = super().reset()
            except RuntimeError as exc:
                if str(exc) != 'Could not sample a reachable, visible, collision-free initial pose.':
                    raise
                continue
            self.initial_z = self._obj_pos()[2]
            return obs
        raise RuntimeError('Could not sample a reachable, visible, collision-free initial pose.')
