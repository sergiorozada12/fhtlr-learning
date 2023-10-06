import random
import numpy as np
import torch
from multiprocessing import Pool

from src.environments import WirelessCommunicationsEnv
from src.utils import Discretizer, ReplayBuffer

torch.set_num_threads(1)


def greedy_episode_nn(env, Q, H, bucket_a, discretizer):
    with torch.no_grad():
        G = 0
        (g, occ, q, b), _ = env.reset()
        s = torch.tensor(np.concatenate([g, occ, [q, b]]))
        for h in range(H):
            a_idx_flat = Q(s).argmax().detach().item()
            a_idx = np.unravel_index(a_idx_flat, bucket_a)
            a = discretizer.get_action_from_index(a_idx)

            (g, occ, q, b), r, d, _, _ = env.step(a)
            s = torch.tensor(np.concatenate([g, occ, [q, b]]))

            G += r

            if d:
                break
        return G


def select_action(Q, s, epsilon, discretizer, bucket_a):
    if np.random.rand() < epsilon:
        a_idx = tuple(np.random.randint(bucket_a).tolist())
        a_idx_flat = np.ravel_multi_index(a_idx, bucket_a)
        a = discretizer.get_action_from_index(a_idx)
        return a, a_idx_flat
    a_idx_flat = Q(s).argmax().detach().item()
    a_idx = np.unravel_index(a_idx_flat, bucket_a)
    a = discretizer.get_action_from_index(a_idx)
    return a, a_idx_flat


def run_experiment(n):
    random.seed(n)
    np.random.seed(n)
    torch.manual_seed(n)

    E = 100_000
    H = 5
    lr = 0.001
    eps = 1.0
    eps_decay = 0.9999
    eps_min = 0.0
    gamma = 0.99
    buffer = ReplayBuffer(1_000)

    Gs = []
    Q = ValueNetwork(6, [32], 100).double()
    opt = torch.optim.Adam(Q.parameters(), lr=lr)
    for episode in range(E):
        (g, occ, q, b), _ = env.reset()
        s = torch.tensor(np.concatenate([g, occ, [q, b]]))
        for h in range(H):
            a, a_idx = select_action(Q, s, eps, discretizer, bucket_a)
            (g, occ, q, b), r, d, _, _ = env.step(a)
            sp = torch.tensor(np.concatenate([g, occ, [q, b]]))

            buffer.append(s, a_idx, sp, r, d)

            st, at, spt, rt, dt = buffer.sample(1)
            q_target = rt + gamma * Q(spt).max(dim=1).values.detach()
            q_hat = Q(st)[torch.arange(at.shape[0]), at]

            opt.zero_grad()
            loss = torch.nn.MSELoss()
            loss(q_hat, q_target).backward()
            opt.step()

            if d:
                break

            s = sp
            eps = max(eps * eps_decay, eps_min)
        G = greedy_episode_nn(env, Q, H, bucket_a, discretizer)
        Gs.append(G)
        # if episode % 5_000 == 0:
        # print(episode, G, flush=True)
    print(n, np.mean(Gs[-1000:]), flush=True)
    return Gs


env = WirelessCommunicationsEnv(
    T=5,  # Number of time slots
    K=2,  # Number of channels
    snr_max=6,  # Max SNR
    snr_min=2,  # Min SNR
    snr_autocorr=0.7,  # Autocorrelation coefficient of SNR
    P_occ=np.array(
        [
            [0.5, 0.3],
            [0.5, 0.7],
        ]
    ),
    occ_initial=[1, 1],  # Initial occupancy state
    batt_harvest=1,  # Battery to harvest following a Bernoulli
    P_harvest=0.4,  # Probability of harvest energy
    batt_initial=5,  # Initial battery
    batt_max_capacity=50,  # Maximum capacity of the battery
    batt_weight=1.0,  # Weight for the reward function
    queue_initial=5,  # Initial size of the queue
    queue_weight=2.0,  # Weight for the reward function
    loss_busy=0.5,  # Loss in the channel when busy
)

bucket_s = [10, 10, 2, 2, 20, 20]
bucket_a = [10, 10]

discretizer = Discretizer(
    min_points_states=[0, 0, 0, 0, 0, 0],
    max_points_states=[10, 10, 1, 1, 5, 5],
    bucket_states=bucket_s,
    min_points_actions=[0, 0],
    max_points_actions=[2, 2],
    bucket_actions=bucket_a,
)

n_exp = 100
if __name__ == "__main__":
    with Pool() as pool:
        results = pool.map(run_experiment, range(n_exp))
    np.save("dqn3.npy", results)
