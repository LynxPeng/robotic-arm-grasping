"""Compact SAC with HIL-SERL/RLPD-style demo replay (50/50 batch sampling).

Only torch + numpy required. Twin Q critics, tanh-squashed Gaussian actor,
automatic temperature tuning, soft target updates. The replay batch mixes
half demonstration transitions and half online transitions every update.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(sizes):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = mlp([obs_dim, hidden, hidden])
        self.mean = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)

    def forward(self, obs):
        h = self.net(obs)
        return self.mean(h), torch.clamp(self.log_std(h), -5.0, 2.0)

    def sample(self, obs):
        mean, log_std = self(obs)
        normal = torch.distributions.Normal(mean, log_std.exp())
        x = normal.rsample()
        a = torch.tanh(x)
        logp = normal.log_prob(x) - torch.log(1 - a.pow(2) + 1e-6)
        return a, logp.sum(-1, keepdim=True)

    def deterministic(self, obs):
        return torch.tanh(self(obs)[0])


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.q1 = mlp([obs_dim + act_dim, hidden, hidden, 1])
        self.q2 = mlp([obs_dim + act_dim, hidden, hidden, 1])

    def forward(self, obs, act):
        x = torch.cat([obs, act], -1)
        return self.q1(x), self.q2(x)


class ReplayBuffer:
    def __init__(self, capacity, obs_dim, act_dim):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), np.float32)
        self.act = np.zeros((capacity, act_dim), np.float32)
        self.rew = np.zeros((capacity, 1), np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), np.float32)
        self.done = np.zeros((capacity, 1), np.float32)
        self.size, self.ptr = 0, 0

    def add(self, o, a, r, no, d):
        i = self.ptr
        self.obs[i], self.act[i], self.rew[i, 0], self.next_obs[i], self.done[i, 0] = o, a, r, no, d
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, n, rng):
        idx = rng.integers(0, self.size, n)
        return self.obs[idx], self.act[idx], self.rew[idx], self.next_obs[idx], self.done[idx]


class SAC:
    def __init__(self, obs_dim, act_dim, device="cpu", lr=3e-4, gamma=0.99, tau=0.005,
                 batch_size=256, demo_ratio=0.5, bc_coef=30.0, buffer_size=300_000,
                 hidden=256, seed=0):
        self.device = torch.device(device)
        self.gamma, self.tau = gamma, tau
        self.batch_size, self.demo_ratio, self.bc_coef = batch_size, demo_ratio, bc_coef
        self.actor = Actor(obs_dim, act_dim, hidden).to(self.device)
        self.critic = Critic(obs_dim, act_dim, hidden).to(self.device)
        self.critic_target = Critic(obs_dim, act_dim, hidden).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        # Start with a small temperature. A large initial alpha inflates soft-Q
        # with an entropy bonus (~alpha*|logp|/(1-gamma)); the resulting actor
        # gradient destroys a precise BC policy before auto-tuning lowers alpha.
        self.log_alpha = torch.full((1,), float(np.log(0.01)), requires_grad=True,
                                    device=self.device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)
        # Lower entropy target for precise manipulation: keeps alpha small so
        # the entropy term does not erode the BC-anchored policy over training.
        self.target_entropy = -2.0 * float(act_dim)
        self.online = ReplayBuffer(buffer_size, obs_dim, act_dim)
        self.demos = None
        self.rng = np.random.default_rng(seed)
        self.updates = 0

    def load_demos(self, path):
        d = np.load(path)
        self.demos = {k: d[k] for k in ("obs", "act", "rew", "next_obs", "done")}
        return len(self.demos["obs"])

    def bc_update(self, batch_size=256):
        """Auxiliary behavior-cloning step: regress deterministic actions onto demos."""
        idx = self.rng.integers(0, len(self.demos["obs"]), batch_size)
        o = torch.as_tensor(self.demos["obs"][idx], device=self.device)
        a = torch.as_tensor(self.demos["act"][idx], device=self.device)
        loss = F.mse_loss(self.actor.deterministic(o), a)
        self.actor_opt.zero_grad()
        loss.backward()
        self.actor_opt.step()
        return loss.item()

    @property
    def alpha(self):
        return self.log_alpha.exp().item()

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            a = self.actor.deterministic(o) if deterministic else self.actor.sample(o)[0]
        return a.squeeze(0).cpu().numpy()

    def _sample_batch(self):
        n_demo = int(self.batch_size * self.demo_ratio) if self.demos is not None else 0
        n_online = self.batch_size - n_demo
        parts = [self.online.sample(n_online, self.rng)]
        if n_demo:
            idx = self.rng.integers(0, len(self.demos["obs"]), n_demo)
            parts.append(tuple(self.demos[k][idx]
                               for k in ("obs", "act", "rew", "next_obs", "done")))
        return tuple(np.concatenate([p[i] for p in parts], 0) for i in range(5))

    def update(self):
        o, a, r, no, d = [torch.as_tensor(x, device=self.device) for x in self._sample_batch()]
        with torch.no_grad():
            a2, logp2 = self.actor.sample(no)
            q1t, q2t = self.critic_target(no, a2)
            y = r + self.gamma * (1 - d) * (torch.min(q1t, q2t) - self.alpha * logp2)
        q1, q2 = self.critic(o, a)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        a_pi, logp = self.actor.sample(o)
        q1p, q2p = self.critic(o, a_pi)
        q_pi = torch.min(q1p, q2p)
        actor_loss = (self.alpha * logp - q_pi).mean()
        if self.bc_coef > 0 and self.demos is not None:
            # Fixed-weight BC anchor. Do NOT normalize by mean|Q|: that ratio
            # explodes while the critic is still random early in training and
            # destroys the actor. A fixed weight keeps the policy near demos.
            idx = self.rng.integers(0, len(self.demos["obs"]), o.shape[0])
            do = torch.as_tensor(self.demos["obs"][idx], device=self.device)
            da = torch.as_tensor(self.demos["act"][idx], device=self.device)
            actor_loss = actor_loss + self.bc_coef * F.mse_loss(self.actor.deterministic(do), da)
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        with torch.no_grad():
            for tp, cp in zip(self.critic_target.parameters(), self.critic.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * cp.data)
        self.updates += 1
        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss.item(),
                "alpha": self.alpha, "alpha_loss": alpha_loss.item()}