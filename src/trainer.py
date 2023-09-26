import random
import numpy as np
import torch
from gym import Env


torch.set_num_threads(1)


def run_experiment(
        n: int,
        E: int,
        H: int,
        eps: float,
        eps_decay: float,
        env: Env,
        agent
    ):
    random.seed(n)
    np.random.seed(n)
    torch.manual_seed(n)

    Gs = []
    for _ in range(E):
        s, _ = env.reset()
        for h in range(H):
            a = agent.select_action(h, s, eps)
            sp, r, d, _, _ = env.step(a)

            agent.buffer.append(h, s, a, sp, r, d)
            agent.update()

            if d:
                break

            s = sp
            eps *= eps_decay
        G = agent.greedy_episode(env, H)
        Gs.append(G)
    return Gs
