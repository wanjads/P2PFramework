import numpy as np
import parameters

"""
Provides constants for the simulation.
For documentation on those variables see documentation in tabular.py, env.py and main.py.
"""

# q-learning
learning_rate = 7e-3
aoi_cap = 100

# epsilon greedy
epsilon_init = 0.9
epsilon_end = 0.0001
delta = (epsilon_end / epsilon_init) ** (1 / parameters.train_eps)

# energy-harvesting
if parameters.energy_harvesting == "unlimited":
    battery_start = parameters.B_max
else:
    battery_start = 0

a0 = 2
a1 = -3
def beta(battery): return a0 * np.exp(a1 * (battery / parameters.B_max))



# random agent
# the probability is matched to the average energy income.
# The random agent takes action a = 3 with this probability, else a = 0.
if parameters.energy_harvesting == "harvested":
    if parameters.sensing == "active":
        p_random = parameters.h_max / (2 * (parameters.send_energy + parameters.sense_energy))
    else:
        p_random = parameters.h_max / (2 * parameters.send_energy)
elif parameters.energy_harvesting == "constrained":
    if parameters.sensing == "active":
        p_random = parameters.budget / (parameters.send_energy + parameters.sense_energy)
    else:
        p_random = parameters.budget / parameters.send_energy
elif parameters.energy_harvesting == "unlimited":
    p_random = parameters.new_package_prob


# action mapping
# (sensing, send)
# maps action value to 0/1 for sensing and sending
action = {
    0: (0, 0),
    1: (0, 1),
    2: (1, 0),
    3: (1, 1)
}

# plotting-constants
No_battery_history = 2
battery_history_length = 2000
use_latex = True
No_cost_history = 3

# priming constants
prime_value_low = 0
prime_value_high = 30
obs_shape = (1, 3)

# moving average constants
ma_window = 20000
