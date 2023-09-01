import numpy as np

# SETUP DESCRIPTION
# - Wireless communication setup, focus in one user sharing the media with other users
# - Finite horizon transmission (T slots)
# - User is equipped with a battery and queue where info is stored
# - K orthogonal channels, user can select power for each time instant
# - Rate given by Shannon's capacity formula

# STATES
# - Amount of energy in the battery: bt (real and positive)
# - Amount of packets in the queue: queuet (real and positive)
# - Normalized SNR (aka channel gain) for each of the channels: gkt (real and positive)
# - Channel being currently occupied: ok (binary)

# ACTIONS
# - Accessing or not each of the K channels pkt
# - Tx power for each of the K channels
# - We can merge both (by setting pkt=0)


class WirelessCommunicationsEnv:
    def __init__(
        self,
        T=10,                         # Number of time slots
        K=3,                          # Number of channels
        snr_max=10,                   # Max SNR
        snr_min=2,                    # Min SNR
        snr_autocorr=0.7,             # Autocorrelation coefficient of SNR
        P_occ=np.array([              # Prob. of transition of occupancy
            [0.3, 0.5],
            [0.7, 0.5],
        ]),
        occ_initial=[1, 1, 1],        # Initial occupancy state 
        batt_harvest=3,               # Battery to harvest following a Bernoulli
        P_harvest=0.5,                # Probability of harvest energy
        batt_initial=5,               # Initial battery
        batt_max_capacity=50,         # Maximum capacity of the battery
        batt_weight=1.0,              # Weight for the reward function
        queue_initial=20,             # Initial size of the queue
        queue_weight=1e-1,            # Weight for the reward function
        loss_busy=0.80,               # Loss in the channel when busy
    ):
        self.T = T
        self.K = K

        self.snr = np.linspace(snr_max, snr_min, K)
        self.snr_autocorr = snr_autocorr

        self.occ_initial = occ_initial
        self.P_occ = P_occ

        self.batt_harvest = batt_harvest
        self.batt_initial = batt_initial
        self.P_harvest = P_harvest
        self.batt_max_capacity = batt_max_capacity
        self.batt_weight = batt_weight

        self.queue_initial = queue_initial
        self.queue_weight = queue_weight

        self.loss_busy = loss_busy

    def step(self, p):
        if np.sum(p) > self.batt[self.t]:
            p = self.batt[self.t] * p / np.sum(p)

        self.c[:, self.t] = np.log2(1 + self.g[:, self.t] * p)
        self.c[:, self.t] *= (1 - self.loss_busy)*self.occ[:, self.t] + (1 - self.occ[:, self.t])

        self.t += 1

        self.h[:, self.t] = np.sqrt(0.5 * self.snr) * (np.random.randn(self.K) + 1j * np.random.randn(self.K))
        self.h[:, self.t] *= np.sqrt(1 - self.snr_autocorr)
        self.h[:, self.t] += np.sqrt(self.snr_autocorr) * self.h[:, self.t - 1]
        self.g[:, self.t] = np.abs(self.h[:, self.t])**2 

        self.occ[:, self.t] += (np.random.rand(self.K) > self.P_occ[1, 1]) * self.occ[:, self.t - 1]
        self.occ[:, self.t] += (np.random.rand(self.K) > self.P_occ[0, 0]) * (1 - self.occ[:, self.t - 1])

        energy_harv = self.batt_harvest * (self.P_harvest > np.random.rand())
        self.batt[self.t] = self.batt[self.t - 1] - np.sum(p) + energy_harv
        self.batt[self.t] = np.clip(self.batt[self.t], 0, self.batt_max_capacity)

        packets = 0
        if self.batt[self.t - 1] > 0:
            packets = np.sum(self.c[:, self.t - 1])
        self.queue[self.t] = self.queue[self.t - 1] - packets
        self.queue[self.t] = max(0, self.queue[self.t])

        r = self.batt_weight * np.log(1 + self.batt[self.t]) - self.queue_weight * self.queue[self.t]
        done = True if self.t == self.T else False
        
        return self._get_obs(self.t), r, done, None, None

    def reset(self):
        self.t = 0
        self.h = np.zeros((self.K, self.T + 1), dtype=np.complex64)
        self.g = np.zeros((self.K, self.T + 1))
        self.c = np.zeros((self.K, self.T + 1))
        self.occ = np.zeros((self.K, self.T + 1))
        self.queue = np.zeros(self.T + 1)
        self.batt = np.zeros(self.T + 1)

        self.h[:, 0] = np.sqrt(0.5 * self.snr) * (np.random.randn(self.K) + 1j * np.random.randn(self.K))
        self.g[:, 0] = np.abs(self.h[:, 0])**2
        self.occ[:, 0] = self.occ_initial
        self.queue[0] = self.queue_initial
        self.batt[0] = self.batt_initial

        return self._get_obs(0), None

    def _get_obs(self, t):
        return (self.g[:, t], self.occ[:, t], self.queue[t], self.batt[t])