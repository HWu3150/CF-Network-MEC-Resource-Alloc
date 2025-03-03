# ----------------------- env params -----------------------------
NUM_MDS = 2
NUM_APS = 2

# ----------------------- action space ---------------------------
DISCRETE_POWERS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # action space (discrete)
P_MAX = 0.8 # action space (continuous)

# ----------------------- channel state --------------------------
SMALL_SCALE_FADING = [0.3, 0.8]
BANDWIDTH = 0.5e6
CHANNEL_NOISE = 2.5e-12
G = -90  # fixed

# ----------------------- CU states ------------------------------
CPU_CAPACITY = 1e9
CYCLES_PER_BIT = 1000

# ----------------------- data to send ---------------------------
DATA_SIZE = 1e6  # assume all tasks are of this size

# ----------------------- time step length -----------------------
t = 0.001
