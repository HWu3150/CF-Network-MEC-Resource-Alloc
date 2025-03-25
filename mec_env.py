from datetime import datetime
import os

import numpy as np
import logging


def compute_sinr_iter(powers, h_mk, G_mk, channel_noise):
    """
    Compute SINR
    :param powers: [p_m1, p_m2]
    :param h_mk: [[h_m1_k1, h_m1_k2], [h_m2_k1, h_m2_k2]], shape (M, K)
    :param G_mk: [[G_m1_k1, G_m1_k2], [G_m2_k1, G_m2_k2]], shape (M, K)
    :param channel_noise: scalar
    :return: sinr for each device, shape (M,)
    """
    num_mds = len(powers)
    num_aps = len(h_mk[0])
    sinrs = np.zeros(num_mds)

    for m in range(num_mds):
        signal_power = powers[m] * sum(G_mk[m][ap] * abs(h_mk[m][ap]) ** 2 for ap in range(num_aps)) ** 2
        interference = 0
        for n in range(num_mds):
            if n != m:
                interference += powers[n] * sum(
                    np.sqrt(G_mk[n][k] * G_mk[m][k]) * h_mk[n][k] * h_mk[m][k] for k in range(num_aps)) ** 2
        sinrs[m] = signal_power / (interference + channel_noise)
    return sinrs


def compute_sinr(powers, h_mk, G_mk, channel_noise):
    """
    Compute SINR.
    :param powers: [p_m1, p_m2]
    :param h_mk: [[h_m1_k1, h_m1_k2], [h_m2_k1, h_m2_k2]], shape (M, K)
    :param G_mk: [[G_m1_k1, G_m1_k2], [G_m2_k1, G_m2_k2]], shape (M, K)
    :param channel_noise: scalar
    :return: sinr for each device, shape (M,)
    """
    # Compute signal power
    signal_power = powers * np.square(np.sum(G_mk * np.abs(h_mk) ** 2, axis=1))

    # Compute interference
    sqrt_G_prod = np.sqrt(G_mk[:, None, :] * G_mk[None, :, :])  # Shape (M, M, K)
    h_prod = h_mk[:, None, :] * h_mk[None, :, :].conj()  # Shape (M, M, K)
    interference_matrix = np.square(np.sum(sqrt_G_prod * h_prod, axis=2))  # Shape (M, M)

    interference = np.dot(powers, interference_matrix) - powers * np.diag(interference_matrix)

    # Compute SINR
    sinr_values = signal_power / (interference + channel_noise)

    return sinr_values


def compute_transmission_rates(sinrs, bandwidth):
    """
    Compute transmission rates.
    :param sinrs: SINR of each MD.
    :param bandwidth: Bandwidth of the channel.
    :return: Transmission rates of each MD.
    """
    return bandwidth * np.log2(1 + sinrs)


class MECEnv:
    def __init__(self, num_mds, num_aps, data_size,
                 small_scale_fading_values, bandwidth,
                 channel_noise, large_scale_fading,
                 discrete_powers, t_length):
        # ------------------------------- invariants ---------------------------------------------
        self.channel_state_trans_prob = np.array([               # Markov state transition matrix
            [0.8, 0.2],     # P[h_idx=0 -> (0,1)]
            [0.3, 0.7],     # P[h_idx=1 -> (0,1)]
        ])
        self.num_mds = num_mds
        self.num_aps = num_aps
        self.data_size = data_size
        self.h_values = np.array(small_scale_fading_values)     # Possible small scale fading values
        self.bandwidth = bandwidth
        self.channel_noise = channel_noise
        self.G = large_scale_fading
        self.discrete_powers = discrete_powers
        self.t_length = t_length

        # ------------------------------- variants ------------------------------------------------
        self.time_step = 0
        # Small scale fading (indices)
        self.h_idx = np.random.randint(0, self.h_values.shape[0], size=(self.num_mds, self.num_aps))
        self.h_mk = np.array([
            [self.h_values[self.h_idx[m, k]] for k in range(self.num_aps)]
            for m in range(self.num_mds)
        ])                                                              # [[h11, h12], [h21, h22], ...]
        self.G_mk = np.full((self.num_mds, self.num_aps,), self.G)           # [[G11, G12], [G21, G22], ...]
        self.d_md = np.full(self.num_mds, self.data_size, dtype=float)     # [d1, d2, ...]
        self.d_md_percent = np.ones(self.num_mds)
        self.aoi_md = np.zeros(self.num_mds)

        # Log configs
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        log_dir = os.path.join(project_root, "logs")
        print(log_dir)
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = os.path.join(log_dir, f"mec_env_{timestamp}.log")
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format="%(message)s",
            filemode="w"
        )

    def reset(self):
        """
        Reset environment
        :return: current state
        """
        self.h_idx = np.random.randint(0, self.h_values.shape[0], size=(self.num_mds, self.num_aps))
        # Real small scale fading values
        self.h_mk = np.array([
            [self.h_values[self.h_idx[m, k]] for k in range(self.num_aps)]
            for m in range(self.num_mds)
        ])  # [[h11, h12], [h21, h22], ...]
        self.G_mk = np.full((self.num_mds, self.num_aps,), self.G)  # [[G11, G12], [G21, G22], ...]
        self.d_md = np.full(self.num_mds, self.data_size, dtype=float)  # [d1, d2, ...]
        self.d_md_percent = np.ones(self.num_mds)
        # self.d_cu = np.zeros(NUM_MDS) # initial # of bits at CU
        # self.cpu_cycles = np.full(NUM_MDS, DATA_SIZE * CYCLES_PER_BIT)
        self.time_step = 0
        self.aoi_md = np.zeros(self.num_mds)
        return self._get_state()

    def get_env_params(self):
        return self.num_mds, self.discrete_powers

    def _get_state(self):
        # Get current state
        # return np.concatenate([self.h_mk.flatten(),
        #                        self.d_md.flatten(),
        #                        self.d_cu.flatten(),
        #                        self.cpu_cycles.flatten()])
        # return tuple(np.concatenate((self.d_md_percent, self.h_mk.ravel(), self.aoi_md)))
        return tuple(np.concatenate((self.d_md_percent, self.h_mk.ravel())))

    def _update_channel_state(self):
        """
        Update channel state
        """
        # sample next state
        for m in range(self.num_mds):
            for k in range(self.num_aps):
                current_state = self.h_idx[m, k]
                # pick next state from [0, 1]
                next_state = np.random.choice(
                    [0, 1],
                    p=self.channel_state_trans_prob[current_state]
                )
                self.h_idx[m, k] = next_state

        # update self.h_mk
        for m in range(self.num_mds):
            for k in range(self.num_aps):
                self.h_mk[m, k] = self.h_values[self.h_idx[m, k]]

    # def quantize_transmitted_bits(self, value, delta=50):
    #     return np.round(value / delta) * delta

    def step(self, actions):
        self._update_channel_state()
        previous_max_percent = np.max(self.d_md_percent)
        # Compute SINR
        sinrs = compute_sinr(actions, self.h_mk, self.G_mk, self.channel_noise)
        # Compute transmission rate
        rates = compute_transmission_rates(sinrs, self.bandwidth)
        # print(rates)

        # Update transmission
        for m in range(self.num_mds):
            # Quantize
            # transmitted = rates[m] * self.t_length
            transmitted = rates[m] * self.t_length
            self.d_md[m] = max(0., self.d_md[m] - transmitted)
            # self.d_cu[m] += transmitted

            # MD can transmit a new data packet
            # if self.d_md[m] == 0:
            #     self.aoi_md[m] = 0
            #     self.d_md[m] = self.data_size
            # else:
            #     self.aoi_md[m] += 1

        self.d_md_percent = self.d_md / self.data_size
        current_max_percent = np.max(self.d_md_percent)
        # print(self.d_md_percent)

        # compute_time = np.max(self.cpu_cycles / CPU_CAPACITY)
        # self.cpu_cycles = np.maximum(0, self.cpu_cycles - CPU_CAPACITY * t)
        self.time_step += 1

        # =========================== Reward function ===========================
        # reward = -self.t_length * self.time_step
        # reward = -np.sum(self.aoi_md)

        # reward = 10 * (previous_max_percent - current_max_percent)
        # reward -= 0.1 * np.sum(self.aoi_md)
        reward = -1


        # =========================== Terminal condition ========================
        # done = np.all(self.d_md == 0) and np.all(self.cpu_cycles == 0)
        # if np.all(self.d_md == 0.):
        #     reward += 1.0
        # done = self.time_step >= 50
        done = np.all(self.d_md == 0)
        logging.info(f"Step {self.time_step}: Actions = {actions}, State = {self.d_md}{self.h_mk}, Reward = {reward}")

        return self._get_state(), reward, done
