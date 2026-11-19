"""Continuous action Q networks and batched cross-entropy search."""

import torch
from torch import nn


class GraspQ(nn.Module):
    def __init__(self, obs_dim, width, scale_observations=False):
        super().__init__()
        # Emphasize centimeter-scale alignment errors and lift in the Panda observation.
        center = (0.5, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        scale = (4.0, 6.0, 4.0, 1.0, 10.0, 10.0, 4.0, 8.0, 1.0, 1.0, 1.0)
        self.register_buffer('obs_center', torch.tensor(center) if scale_observations else None,
                             persistent=False)
        self.register_buffer('obs_scale', torch.tensor(scale) if scale_observations else None,
                             persistent=False)
        self.state = nn.Linear(obs_dim, width)
        self.action = nn.Linear(4, width, bias=False)
        self.head = nn.Sequential(nn.SiLU(), nn.Linear(width, width), nn.SiLU(),
                                  nn.Linear(width, 1))

    def forward(self, observation, action):
        if self.obs_scale is not None:
            observation = (observation - self.obs_center) * self.obs_scale
        state = self.state(observation)
        if action.ndim == 3:
            state = state.unsqueeze(1)
        return self.head(state + self.action(action)).squeeze(-1)


@torch.no_grad()
def cem_action(score, obs, population=96, iterations=3, action_dim=4, center=None, radius=None):
    """Search bounded Cartesian increments and a continuous gripper setpoint."""
    batch = obs.shape[0]
    mean = obs.new_zeros(batch, 1, action_dim)
    std = torch.ones_like(mean)
    low, high = -1, 1
    if center is not None:
        mean = center[:, None].to(dtype=obs.dtype)
        std = torch.ones_like(mean) * radius / 2
        low, high = (mean-radius).clamp(-1, 1), (mean+radius).clamp(-1, 1)
    best = mean
    elite_count = max(2, population // 8)
    for _ in range(iterations):
        noise = torch.randn(batch, population - 1, action_dim, device=obs.device)
        candidates = torch.cat((best, (mean + std * noise).clamp(low, high)), dim=1)
        scores = score(obs, candidates)
        indices = scores.topk(elite_count, dim=1).indices
        elite = candidates.gather(1, indices.unsqueeze(-1).expand(-1, -1, action_dim))
        best = elite[:, :1]
        mean = elite.mean(dim=1, keepdim=True)
        std = elite.std(dim=1, keepdim=True, unbiased=False).clamp_min(0.035 if center is not None else 0.08)
    return best.squeeze(1)


def td_target(reward, terminal, next_q, discount):
    return reward + discount * torch.where(terminal, 0., next_q)


class Replay:
    def __init__(self, capacity, size, device):
        self.data = torch.empty(capacity, size, dtype=torch.float32, device=device)
        self.count = self.cursor = 0

    def extend(self, rows):
        capacity = len(self.data)
        rows = rows[-capacity:]
        end = min(capacity - self.cursor, len(rows))
        self.data[self.cursor:self.cursor + end].copy_(rows[:end])
        self.data[:len(rows) - end].copy_(rows[end:])
        self.cursor = (self.cursor + len(rows)) % capacity
        self.count = min(capacity, self.count + len(rows))

    def sample(self, size):
        indices = torch.randint(self.count, (size,), device=self.data.device)
        return self.data[indices]
