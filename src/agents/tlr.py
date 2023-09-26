import numpy as np

from torch import no_grad
from torch.nn import MSELoss
from torch.optim import Adam

from src.models import PARAFAC
from src.utils import ReplayBuffer, Discretizer


class FHTlr:
    def __init__(
            self,
            discretizer: Discretizer,
            alpha: float,
            H: int,
            k: int,
            scale: float
        ) -> None:
        self.alpha = alpha
        self.H = H

        self.buffer = ReplayBuffer(1)
        self.discretizer = discretizer
        self.Q = PARAFAC(
            [H] + discretizer.bucket_s + discretizer.bucket_a,
            k=k,
            scale=scale
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
        s_idx = [h] + self.discretizer.get_state_index(s)
        a_idx_flat = self.Q(s_idx).argmax().detach().item()
        a_idx = np.unravel_index(a_idx_flat, self.discretizer.bucket_a)
        a = self.discretizer.get_action_from_index(a_idx)
        return a

    def update(self) -> None:
        h, s, a, sp, r, d = self.buffer.sample(1)

        s_idx = [h] + self.discretizer.get_state_index(s)
        sp_idx = [h + 1] + self.discretizer.get_state_index(sp)
        a_idx = self.discretizer.get_action_index(a)

        self.opt.zero_grad()
        loss = MSELoss()
        loss(q_hat, q_target).backward()
        self.opt.step()

        for factor in self.Q.factors:
            q_target = r
            if not d:
                q_target += self.Q(sp_idx).max().detach()
            q_hat = self.Q(s_idx + a_idx)
        
            self.opt.zero_grad()
            loss = MSELoss()
            loss(q_hat, q_target).backward()

            with no_grad():
                for frozen_factor in self.Q.factors:
                    if frozen_factor is not factor:
                        frozen_factor.grad = None
            self.opt.step()
