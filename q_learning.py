import random
import numpy as np
from multiprocessing import Pool

from src.environments import WirelessCommunicationsEnv
from src.utils import Discretizer, ReplayBuffer


def greedy_episode_ql(env, Q, H, bucket_a, discretizer):
    G = 0
    (g, occ, q, b), _ = env.reset()
    s = np.concatenate([[0], g, occ, [q, b]])
    s_idx = discretizer.get_state_index(s)
    for h in range(H):
        q = Q[s_idx]
        idx_flat = q.argmax()
        a_idx = np.unravel_index(idx_flat, q.shape)
        a = discretizer.get_action_from_index(a_idx)

        (g, occ, q, b), r, d, _, _ = env.step(a)
        s = np.concatenate([[h + 1], g, occ, [q, b]])
        s_idx = discretizer.get_state_index(s)
        
        G += r
        
        if d:
            break
    return G

def select_action(Q, s_idx, epsilon):
    if np.random.rand() < epsilon:
        a_idx = tuple(np.random.randint(bucket_a).tolist())
        a = discretizer.get_action_from_index(a_idx)
        return a, a_idx
    q = Q[s_idx]
    idx_flat = q.argmax()
    a_idx = np.unravel_index(idx_flat, q.shape)
    a = discretizer.get_action_from_index(a_idx)
    return a, a_idx

def run_experiment(n):
    random.seed(n)
    np.random.seed(n)

    E = 100_000
    H = 5
    lr = 1.0
    eps = 1.0
    eps_decay = 0.9999
    eps_min = 0.0

    Gs = []
    Q = np.zeros(bucket_s + bucket_a)
    for episode in range(E):
        (g, occ, q, b), _ = env.reset()
        s = np.concatenate([[0], g, occ, [q, b]])
        s_idx = discretizer.get_state_index(s)
        for h in range(H):
            a, a_idx = select_action(Q, s_idx, eps)
            (g, occ, q, b), r, d, _, _ = env.step(a)
            sp = np.concatenate([[h + 1], g, occ, [q, b]])
            sp_idx = discretizer.get_state_index(sp)
            
            m = -1*(d - 1)
            q_target = r + m * Q[sp_idx].max()
            q_hat = Q[s_idx + a_idx]
            
            Q[s_idx + a_idx] += lr*(q_target - q_hat)
            
            if d:
                break
                
            s = sp
            s_idx = sp_idx
            eps = max(eps*eps_decay, eps_min)
        G = greedy_episode_ql(env, Q, H, bucket_a, discretizer)
        Gs.append(G)
        # if episode % 5_000 == 0:
            # print(episode, G)
    print(n, np.mean(Gs[-1000:]), flush=True)
    return Gs

env = WirelessCommunicationsEnv(
    T=5,                          # Number of time slots
    K=2,                          # Number of channels
    snr_max=6,                    # Max SNR
    snr_min=2,                    # Min SNR
    snr_autocorr=0.7,             # Autocorrelation coefficient of SNR
    P_occ=np.array([
        [0.5, 0.3],
        [0.5, 0.7],
    ]),
    occ_initial=[1, 1],           # Initial occupancy state 
    batt_harvest=1,               # Battery to harvest following a Bernoulli
    P_harvest=0.4,                # Probability of harvest energy
    batt_initial=5,               # Initial battery
    batt_max_capacity=50,         # Maximum capacity of the battery
    batt_weight=1.0,              # Weight for the reward function
    queue_initial=5,             # Initial size of the queue
    queue_weight=2.0,             # Weight for the reward function
    loss_busy=0.5,                # Loss in the channel when busy
)

bucket_s = [6, 10, 10, 2, 2, 10, 10]
bucket_a = [10, 10]

discretizer = Discretizer(
    min_points_states=[0, 0, 0, 0, 0, 0, 0],
    max_points_states=[5, 10, 10, 1, 1, 5, 5],
    bucket_states=bucket_s,
    min_points_actions=[0, 0],
    max_points_actions=[2, 2],
    bucket_actions=bucket_a,
)

n_exp = 100
if __name__ == "__main__":
    with Pool() as pool:
        results = pool.map(run_experiment, range(n_exp))
    np.save('ql.npy', results)
