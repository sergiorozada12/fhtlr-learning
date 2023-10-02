from src.environments import GridWorldEnv
from src.agents.ql import QLearning, FHQLearning
from src.agents.tlr import FHTlr
from src.utils import Discretizer
from src.trainer import run_experiment


ALPHA_Q = .9
ALPHA_TLR = .001
GAMMA = .99
E = 10_000
H = 5
EPS = 1.0
EPS_DECAY = .9999
K = 8
SCALE = .1

ENV = GridWorldEnv()

DISCRETIZER = Discretizer(
    min_points_states=[0, 0],
    max_points_states=[4, 4],
    bucket_states=[5, 5],
    min_points_actions=[0],
    max_points_actions=[3],
    bucket_actions=[4]
)

def run_gridworld_simulations():
    q_learner = QLearning(DISCRETIZER, ALPHA_Q, GAMMA)
    fhq_learner = FHQLearning(DISCRETIZER, ALPHA_Q, H)
    fhtlr_learner = FHTlr(DISCRETIZER, ALPHA_TLR, H, K, SCALE)

    #_ = run_experiment(0, E, H, EPS, EPS_DECAY, ENV, q_learner)
    #_ = run_experiment(0, E, H, EPS, EPS_DECAY, ENV, fhq_learner)
    _ = run_experiment(0, E, H, EPS, EPS_DECAY, ENV, fhq_learner)
