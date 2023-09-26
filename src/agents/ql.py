import numpy as np
from src.utils import ReplayBuffer, Discretizer


class QLearning:
    def __init__(
            self,
            discretizer: Discretizer,
            alpha: float,
            gamma: float
        ) -> None:
        self.alpha = alpha
        self.gamma = gamma

        self.buffer = ReplayBuffer(1)
        self.discretizer = discretizer
        self.Q = np.zeros(discretizer.bucket_s + discretizer.bucket_a)

    def select_action(self, s: np.ndarray, epsilon: float) -> np.ndarray:
        if np.random.rand() < epsilon:
            a_idx = tuple(np.random.randint(self.discretizer.bucket_a).tolist())
            a = self.discretizer.get_action_from_index(a_idx)
            return a
        s_idx = self.discretizer.get_state_index(s)
        q = self.Q[s_idx].argmax()
        a_idx = np.unravel_index(q.argmax(), q.shape)
        a = self.discretizer.get_action_from_index(a_idx)
        return a

    def update(self) -> None:
        _, s, a, sp, r, _ = self.buffer.sample(1)

        s_idx = self.discretizer.get_state_index(s)
        sp_idx = self.discretizer.get_state_index(sp)
        a_idx = self.discretizer.get_action_index(a)

        q_target = r + self.gamma*self.Q[sp_idx].max()
        q_hat = self.Q[s_idx + a_idx]

        self.Q[s_idx + a_idx] += self.alpha*(q_target - q_hat)


class FHQLearning:
    def __init__(
            self,
            discretizer: Discretizer,
            alpha: float,
            H: int
        ) -> None:
        self.alpha = alpha
        self.H = H

        self.buffer = ReplayBuffer(1)
        self.discretizer = discretizer
        self.Q = np.zeros([H] + discretizer.bucket_s + discretizer.bucket_a)

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
        s_idx = self.discretizer.get_state_index(s)
        q = self.Q[[h] + s_idx].argmax()
        a_idx = np.unravel_index(q.argmax(), q.shape)
        a = self.discretizer.get_action_from_index(a_idx)
        return a

    def update(self) -> None:
        h, s, a, sp, r, d = self.buffer.sample(1)

        s_idx = [h] + self.discretizer.get_state_index(s)
        sp_idx = [h + 1] + self.discretizer.get_state_index(sp)
        a_idx = self.discretizer.get_action_index(a)

        q_target = r
        if not d:
            q_target += self.gamma*self.Q[sp_idx].max()
        q_hat = self.Q[s_idx + a_idx]

        self.Q[s_idx + a_idx] += self.alpha*(q_target - q_hat)
