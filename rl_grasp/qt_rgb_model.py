"""Image-conditioned Q model with feature reuse across CEM candidates."""

import numpy as np
import torch
from torch import nn

from qt_rgb_env import SIZE, AUX_DIM, ACT_DIM
from qt_model import cem_action


class ImageReplay:
    """Bounded host replay with uint8 frames and pinned batch staging."""

    def __init__(self, capacity):
        self.capacity = capacity
        shape = capacity, SIZE, SIZE, 3
        self.current = np.empty(shape, dtype=np.uint8)
        self.following = np.empty(shape, dtype=np.uint8)
        self.values = np.empty((capacity, 2*AUX_DIM+ACT_DIM+2), dtype=np.float32)
        self.cursor = self.count = 0

    def append(self, images, aux, actions, next_images, next_aux, rewards, done):
        n = len(images)
        indices = (self.cursor + np.arange(n)) % self.capacity
        self.current[indices] = images
        self.following[indices] = next_images
        self.values[indices] = np.concatenate((aux, actions, next_aux,
                                               rewards[:, None], done[:, None]), axis=1)
        self.cursor = (self.cursor+n) % self.capacity
        self.count = min(self.capacity, self.count+n)

    def sample_into(self, start, stop, batch, rng):
        indices = rng.integers(self.count, size=stop-start)
        for source, dest in zip((self.current, self.following, self.values), batch, strict=True):
            np.take(source, indices, axis=0, out=dest[start:stop].numpy(), mode='clip')


class RGBQ(nn.Module):
    def __init__(self, proposal=False):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 24, 7, stride=4, padding=3), nn.SiLU(),
            nn.Conv2d(24, 40, 5, stride=2, padding=2), nn.SiLU(),
            nn.Conv2d(40, 64, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.SiLU(),
        )
        self.obs = nn.Sequential(nn.Linear(64*7*7+AUX_DIM, 192), nn.LayerNorm(192), nn.SiLU())
        self.proposal = (nn.Sequential(nn.Linear(192, 128), nn.SiLU(),
                                       nn.Linear(128, ACT_DIM), nn.Tanh()) if proposal else None)
        self.register_buffer('radius', torch.tensor([.14, .14, .12, .20, .14]), persistent=False)
        self.action = nn.Linear(ACT_DIM, 192, bias=False)
        self.head = nn.Sequential(nn.SiLU(), nn.Linear(192, 192), nn.SiLU(), nn.Linear(192, 1))

    def encode(self, images, aux):
        image = images.permute(0, 3, 1, 2).to(torch.bfloat16) * (1/255) - 0.5
        visual = self.encoder(image).flatten(1)
        return self.obs(torch.cat((visual, aux.to(visual.dtype)), dim=1))

    def score(self, features, action):
        state = features[:, None] if action.ndim == 3 else features
        return self.head(state + self.action(action)).squeeze(-1)

    def forward(self, image, aux, action):
        return self.score(self.encode(image, aux), action)


def select(model, features, population, iterations):
    center = model.proposal(features) if model.proposal is not None else None
    return cem_action(model.score, features.float(), population=population,
                      iterations=iterations, action_dim=ACT_DIM, center=center, radius=model.radius)
