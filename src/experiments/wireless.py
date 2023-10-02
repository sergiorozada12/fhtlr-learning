import numpy as np

from src.environments import WirelessCommunicationsEnv
from src.agents.ql import QLearning, FHQLearning
from src.agents.dqn import DFHqn, Dqn
from src.agents.tlr import FHTlr
from src.utils import Discretizer
from src.trainer import run_experiment


ALPHA_Q = 1.0
ALPHA_DQN = .001
ALPHA_TLR = .001
BUFFER_SIZE = 1_000
GAMMA = .99
E = 100_000
H = 5
EPS = 1.0
EPS_DECAY = .9999
K = 8
SCALE = .1

ENV = WirelessCommunicationsEnv(
    T=5,
    K=2,
    snr_max=6,
    snr_min=2,
    snr_autocorr=0.7,
    P_occ=np.array([
        [0.5, 0.3],
        [0.5, 0.7],
    ]),
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
    bucket_states=[10, 10, 2, 2, 20, 20],
    min_points_actions=[0, 0],
    max_points_actions=[2, 2],
    bucket_actions=[10, 10],
)

def run_wireless_simulations():
    q_learner = QLearning(DISCRETIZER, ALPHA_Q, GAMMA)
    fhq_learner = FHQLearning(DISCRETIZER, ALPHA_Q, H)
    dqn_learner = Dqn(DISCRETIZER, ALPHA_DQN, GAMMA, BUFFER_SIZE)
    dfhqn_learner = DFHqn(DISCRETIZER, ALPHA_DQN, H, BUFFER_SIZE)
    fhtlr_learner = FHTlr(DISCRETIZER, ALPHA_TLR, H, K, SCALE)
    G = run_experiment(0, E, H, EPS, EPS_DECAY, ENV, fhq_learner)
