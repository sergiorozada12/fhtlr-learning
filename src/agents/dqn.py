import numpy as np

from torch.nn import MSELoss
from torch.optim import Adam

from src.models import ValueNetwork, FHValueNetwork
from src.utils import ReplayBuffer, Discretizer


class Dqn:
    def __init__(
            self,
            discretizer: Discretizer,
            alpha: float,
            gamma: float,
            buffer_size: int
        ) -> None:
        self.gamma = gamma

        self.buffer = ReplayBuffer(buffer_size)
        self.discretizer = discretizer
        self.Q = ValueNetwork(
            len(discretizer.bucket_s),
            [32],
            len(discretizer.bucket_a)
        ).double()
        self.opt = Adam(self.Q.parameters(), lr=alpha)

    def select_action(self, s: np.ndarray, epsilon: float) -> np.ndarray:
        if np.random.rand() < epsilon:
            a_idx = tuple(np.random.randint(self.discretizer.bucket_a).tolist())
            a = self.discretizer.get_action_from_index(a_idx)
            return a
        a_idx_flat = self.Q(s).argmax().detach().item()
        a_idx = np.unravel_index(a_idx_flat, self.discretizer.bucket_a)
        a = self.discretizer.get_action_from_index(a_idx)
        return a

    def update(self) -> None:
        _, s, a, sp, r, _ = self.buffer.sample(1)

        a_idx = self.discretizer.get_action_index(a)

        _, s, a, sp, r, _ = self.buffer.sample(1)
        q_target = r + self.gamma * self.Q(sp).max().detach()
        q_hat = self.Q(s)[a_idx]

        self.opt.zero_grad()
        loss = MSELoss()
        loss(q_hat, q_target).backward()
        self.opt.step()


class DFHqn:
    def __init__(
            self,
            discretizer: Discretizer,
            alpha: float,
            H: int,
            buffer_size: int
        ) -> None:
        self.alpha = alpha
        self.H = H

        self.buffer = ReplayBuffer(buffer_size)
        self.discretizer = discretizer
        self.Q = FHValueNetwork(
            len(discretizer.bucket_s),
            [32],
            len(discretizer.bucket_a)
        ).double()
        self.opt = Adam(self.Q.parameters(), lr=alpha)

    def select_action(
            self,
            s: np.ndarray,
            h: int,
            epsilon: float
        ) -> np.ndarray:
        if np.random.rand() < epsilon:
            a_idx = tuple(np.random.randint(self.discretizer.bucket_a).tolist())
            a = self.discretizer.get_action_from_index(a_idx)
            return a
        a_idx_flat = self.Q(s, h).argmax().detach().item()
        a_idx = np.unravel_index(a_idx_flat, self.discretizer.bucket_a)
        a = self.discretizer.get_action_from_index(a_idx)
        return a

    def update(self) -> None:
        h, s, a, sp, r, d = self.buffer.sample(1)

        a_idx = self.discretizer.get_action_index(a)

        q_target = r
        if not d:
            q_target += self.Q(sp, h + 1).max().detach()
        q_hat = self.Q(s, h)[a_idx]

        self.opt.zero_grad()
        loss = MSELoss()
        loss(q_hat, q_target).backward()
        self.opt.step()
