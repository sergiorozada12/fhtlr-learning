import random
import numpy as np
import torch
from multiprocessing import Pool

from src.environments import WirelessCommunicationsEnv
from src.utils import Discretizer, ReplayBuffer

torch.set_num_threads(1)


class PARAFAC(torch.nn.Module):
    def __init__(self, dims, k, scale=1.0):
        super().__init__()

        self.k = k
        self.n_factors = len(dims)

        factors = []
        for dim in dims:
            factor = scale*torch.randn(dim, k, dtype=torch.double, requires_grad=True)
            factors.append(torch.nn.Parameter(factor))
        self.factors = torch.nn.ParameterList(factors)

    def forward(self, indices):
        prod = torch.ones(self.k, dtype=torch.double)
        for i in range(len(indices)):
            idx = indices[i]
            factor = self.factors[i]
            prod *= factor[idx, :]
        if len(indices) < len(self.factors):
            result = []
            for c1, c2 in zip(self.factors[-2].t(), self.factors[-1].t()):
                kr = torch.kron(c1, c2)
                result.append(kr)
            factors_action = torch.stack(result, dim=1)
            return torch.matmul(prod, factors_action.T)        
        return torch.sum(prod, dim=-1)

def greedy_episode_tlr(env, Q, H, bucket_a, discretizer):
    with torch.no_grad():
        G = 0
        (g, occ, q, b), _ = env.reset()
        s = np.concatenate([g, occ, [q, b]])
        for h in range(H):
            s_idx = tuple([h]) + discretizer.get_state_index(s)
            a_idx_flat = Q(s_idx).argmax().detach().item()
            a_idx = np.unravel_index(a_idx_flat, bucket_a)
            a = discretizer.get_action_from_index(a_idx)
            
            (g, occ, q, b), r, d, _, _ = env.step(a)
            s = np.concatenate([g, occ, [q, b]])
            
            G += r
            
            if d:
                break
        return G

def select_action(Q, s_idx, epsilon, discretizer, bucket_a):
    if np.random.rand() < epsilon:
        a_idx = tuple(np.random.randint(bucket_a).tolist())
        a = discretizer.get_action_from_index(a_idx)
        return a, a_idx
    a_idx_flat = Q(s_idx).argmax().detach().item()
    a_idx = np.unravel_index(a_idx_flat, bucket_a)
    a = discretizer.get_action_from_index(a_idx)
    return a, a_idx

def run_experiment(n):
    random.seed(n)
    np.random.seed(n)
    torch.manual_seed(n)

    E = 100_000
    H = 5
    lr = 0.01
    eps = 1.0
    eps_decay = 0.9999
    eps_min = 0.0
    k = 50

    Gs = []
    Q = PARAFAC([H] + bucket_s + bucket_a, k=k, scale=0.5)
    opt = torch.optim.Adam(Q.parameters(), lr=lr)
    for episode in range(E):
        if episode == 5_000:
            opt = torch.optim.Adam(Q.parameters(), lr=0.001)
        G = 0
        (g, occ, q, b), _ = env.reset()
        s = np.concatenate([g, occ, [q, b]])
        s_idx = tuple([0]) + discretizer.get_state_index(s)
        for h in range(H):
            a, a_idx = select_action(Q, s_idx, eps, discretizer, bucket_a)
            (g, occ, q, b), r, d, _, _ = env.step(a)
            sp = np.concatenate([g, occ, [q, b]])
            sp_idx = tuple([h + 1]) + discretizer.get_state_index(sp)
                    
            G += r
            
            # Update
            for factor in Q.factors:
                q_target = torch.tensor(r).double()
                if not d:
                    q_target += Q(sp_idx).max().detach()
                q_hat = Q(s_idx + a_idx)
            
                opt.zero_grad()
                loss = torch.nn.MSELoss()
                loss(q_hat, q_target).backward()

                with torch.no_grad():
                    for frozen_factor in Q.factors:
                        if frozen_factor is not factor:
                            frozen_factor.grad = None
                opt.step()
            
            if d:
                break
                
            s = sp
            s_idx = sp_idx
            eps = max(eps*eps_decay, eps_min)
        G = greedy_episode_tlr(env, Q, H, bucket_a, discretizer)
        Gs.append(G)
        # if episode % 5_000 == 0:
            # print(episode, G, flush=True)
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

#bucket_s = [10, 10, 2, 2, 20, 20]
bucket_s = [10, 10, 2, 2, 10, 10]
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
    np.save('fhtlr3.npy', results)
