import configs
"""
Provides parameters to the simulation.
Some of those parameters are loaded from the configs.py file based on the config id.
"""

config_id = 2.1
# load config parameters
N_process_states, sensing, energy_harvesting, channel_quality, measure, risk_sensitivity, B_max, h_max, p_remain, \
    send_energy, sense_energy, query_probability, new_package_prob, budget, threshold, risk_factor \
    = configs.load(config_id)

# simulation parameters
sim_only = True  # False or True, False also creates basic plots for the simulation. True recommended for speed.
print_mode = 1  # [0, 1, 2] 0 minimum, 1 everything, 2  minimal updates

train_eps = int(1e5)
test_eps = int(1e4)

num_runs = 100
