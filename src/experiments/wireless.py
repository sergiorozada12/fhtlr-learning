import numpy as np
from multiprocessing import Pool
from functools import partial

from src.environments import WirelessCommunicationsEnv
from src.agents.ql import QLearning, FHQLearning
from src.agents.dqn import DFHqn, Dqn
from src.agents.tlr import FHTlr
from src.utils import Discretizer
from src.trainer import run_experiment
from src.plots import plot_wireless


ALPHA_Q = 1.0
ALPHA_DQN = 0.001
ALPHA_TLR = 0.001
ALPHA_TLR_DECAY_STEP = 5_000
ALPHA_DECAY = 0.1
BUFFER_SIZE = 1_000
GAMMA = 0.99
E = 100_000
H = 5
EPS = 1.0
EPS_DECAY = 0.9999
K = 50
SCALE = 0.5
N_EXPS = 100

ENV = WirelessCommunicationsEnv(
    T=5,
    K=2,
    snr_max=6,
    snr_min=2,
    snr_autocorr=0.7,
    P_occ=np.array(
        [
            [0.5, 0.3],
            [0.5, 0.7],
        ]
    ),
    occ_initial=[1, 1],
    batt_harvest=1,
    P_harvest=0.4,
    batt_initial=5,
    batt_max_capacity=50,
    batt_weight=1.0,
    queue_initial=5,
    queue_weight=2.0,
    loss_busy=0.5,
)


DISCRETIZER = Discretizer(
    min_points_states=[0, 0, 0, 0, 0, 0],
    max_points_states=[10, 10, 1, 1, 5, 5],
    bucket_states=[10, 10, 2, 2, 10, 10],
    min_points_actions=[0, 0],
    max_points_actions=[2, 2],
    bucket_actions=[10, 10],
)


def run_paralell(name: str, agent):
    partial_run = partial(
        run_experiment, E=E, H=H, eps=EPS, eps_decay=EPS_DECAY, env=ENV, agent=agent
    )
    with Pool() as pool:
        results = pool.map(partial_run, range(N_EXPS))
    np.save(f"results/{name}.npy", results)


def run_wireless_simulations():
    q_learner = QLearning(DISCRETIZER, ALPHA_Q, GAMMA)
    fhq_learner = FHQLearning(DISCRETIZER, ALPHA_Q, H)
    dqn_learner = Dqn(DISCRETIZER, ALPHA_DQN, GAMMA, BUFFER_SIZE)
    dfhqn_learner = DFHqn(DISCRETIZER, ALPHA_DQN, H, BUFFER_SIZE)
    fhtlr_learner = FHTlr(
        DISCRETIZER, ALPHA_TLR, H, K, SCALE, ALPHA_DECAY, ALPHA_TLR_DECAY_STEP
    )

    run_paralell('ql2', q_learner)
    run_paralell('dqn2', dqn_learner)
    run_paralell('dfhqn2', dfhqn_learner)
    run_paralell("fhtlr2", fhtlr_learner)

    plot_wireless()
