import constants
import numpy as np
import matplotlib.pyplot as plt

import utils
import configs
"""
Offers multiple plots to visualize simulation data.
Execute the functions with the parameters defining the configuration and the specific simulation to create a plot.
Plots are saved in results/plot/config_id .
Toggle the use of latex (fonts).
"""

# Use Latex font, takes longer to render, requires working latex installation
if constants.use_latex:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Computer Modern"
    })


def plot_bar(config_id, train_eps, test_eps, num_runs):
    """Plots a bar graph comparing the agents tabularQ-TQL, tabularQ-QLF, random and threshold.

    Parameters
    ----------
    config_id : float
        The config_id of the simulation data to plot.
    train_eps : int
        The number of episodes trained.
    test_eps : int
        The number of episodes tested.
    num_runs : int
        The number of runs simulated.

    Returns
    -------
    none

    """
    _, _, _, p, measure, _, _, _, _, _, _, _, new_package_prob, _, _, _ = configs.load(config_id)

    folder_name = '_'.join([str(train_eps), str(test_eps), str(num_runs)])
    try:
        results = np.load('results/' + str(config_id) + '/' + folder_name + '/results.npy')
    except OSError:
        print('No simulation data present for those parameters. Run the simulation and rerun this script.')
        raise OSError

    agent_labels = ['TQL', 'QLF', 'threshold', 'random']

    fig = plt.figure(dpi=100, figsize=(6.4, 4.8))

    plot_costs = results[0:4]
    plot_costs_error = results[8:12]
    plot_risky_states = results[4:8]
    plot_risky_states_error = results[12:16]

    ax1 = fig.add_subplot(2, 1, 1)
    bars1 = ax1.bar(agent_labels, plot_costs, width=0.4)
    ax1.errorbar(agent_labels, plot_costs, plot_costs_error, fmt="none", ecolor='r', capsize=4)
    ax1.set_ylabel('mean' + measure)
    ax1.bar_label(bars1, label_type='center')

    ax2 = fig.add_subplot(2, 1, 2)
    bars2 = ax2.bar(agent_labels, plot_risky_states, width=0.4)
    ax2.errorbar(agent_labels, plot_risky_states, plot_risky_states_error, fmt='none', ecolor='r', capsize=4)
    ax2.set_ylabel('mean risky states')
    ax2.bar_label(bars2, label_type='center')

    fig.tight_layout()
    fig.savefig('results/plot/' + str(config_id) + '/' + folder_name + '_bar.png')


def plot_bar_sensitivity_comparison(config_id, train_eps, test_eps, num_runs, prime_mode):
    """ Plots a bar graph comparing all 4 possible sensitivity modes (energy + risk), the random and the threshold agent.


    Parameters
    ----------
    config_id : float
        The config_id of the simulation data to plot.
    train_eps : int
        The number of episodes trained.
    test_eps : int
        The number of episodes tested.
    num_runs : int
        The number of runs simulated.
    prime_mode : bol
        The mode for priming used.

    Returns
    -------
    none

    """

    _, _, _, p, measure, _, _, _, _, _, _, _, new_package_prob, _, _ = configs.load(config_id)

    folder_name_ff = '_'.join([str(False), str(False),
                               str(train_eps), str(test_eps), str(num_runs),
                               str(prime_mode)])
    folder_name_tf = '_'.join([str(True), str(False),
                               str(train_eps), str(test_eps), str(num_runs),
                               str(prime_mode)])
    folder_name_ft = '_'.join([str(False), str(True),
                               str(train_eps), str(test_eps), str(num_runs),
                               str(prime_mode)])
    folder_name_tt = '_'.join([str(True), str(True),
                               str(train_eps), str(test_eps), str(num_runs),
                               str(prime_mode)])

    file_name = '_'.join(
        [str(train_eps), str(test_eps), str(num_runs), str(prime_mode)])

    try:
        results_ff = np.load('results/' + str(config_id) + '/' + folder_name_ff + '/results.npy')
        results_tf = np.load('results/' + str(config_id) + '/' + folder_name_tf + '/results.npy')
        results_ft = np.load('results/' + str(config_id) + '/' + folder_name_ft + '/results.npy')
        results_tt = np.load('results/' + str(config_id) + '/' + folder_name_tt + '/results.npy')
    except OSError:
        print('No simulation data present for those parameters. Run the simulation and rerun this script.')
        raise OSError

    agent_labels = ['tabularQ:\nrs = False,\nes = False', 'tabularQ:\nrs = True,\nes = False',
                    'tabularQ:\nrs = False,\nes = True', 'tabularQ:\nrs = True,\nes = True', 'threshold', 'random']

    # gather data

    plot_costs_tabular_ff = results_ff[0]
    plot_costs_error_tabular_ff = results_ff[6]
    plot_risky_states_tabular_ff = results_ff[3]
    plot_risky_states_error_tabular_ff = results_ff[9]

    plot_costs_tabular_tf = results_tf[0]
    plot_costs_error_tabular_tf = results_tf[6]
    plot_risky_states_tabular_tf = results_tf[3]
    plot_risky_states_error_tabular_tf = results_tf[9]

    plot_costs_tabular_ft = results_ft[0]
    plot_costs_error_tabular_ft = results_ft[6]
    plot_risky_states_tabular_ft = results_ft[3]
    plot_risky_states_error_tabular_ft = results_ft[9]

    plot_costs_tabular_tt = results_tt[0]
    plot_costs_error_tabular_tt = results_tt[6]
    plot_risky_states_tabular_tt = results_tt[3]
    plot_risky_states_error_tabular_tt = results_tt[9]

    plot_costs_threshold = results_ff[1]
    plot_costs_error_threshold = results_ff[7]
    plot_risky_states_threshold = results_ff[4]
    plot_risky_states_error_threshold = results_ff[10]

    plot_costs_random = results_ff[2]
    plot_costs_error_random = results_ff[8]
    plot_risky_states_random = results_ff[5]
    plot_risky_states_error_random = results_ff[11]

    plot_costs = [plot_costs_tabular_ff, plot_costs_tabular_tf, plot_costs_tabular_ft, plot_costs_tabular_tt,
                  plot_costs_threshold, plot_costs_random]
    plot_costs_error = [plot_costs_error_tabular_ff, plot_costs_error_tabular_tf, plot_costs_error_tabular_ft,
                        plot_costs_error_tabular_tt, plot_costs_error_threshold, plot_costs_error_random]
    plot_risky_states = [plot_risky_states_tabular_ff, plot_risky_states_tabular_tf, plot_risky_states_tabular_ft,
                         plot_risky_states_tabular_tt, plot_risky_states_threshold, plot_risky_states_random]
    plot_risky_states_error = [plot_risky_states_error_tabular_ff, plot_risky_states_error_tabular_tf,
                               plot_risky_states_error_tabular_ft, plot_risky_states_error_tabular_tt,
                               plot_risky_states_error_threshold, plot_risky_states_error_random]

    # plot
    fig = plt.figure(dpi=100, figsize=(10, 4.8))

    ax1 = fig.add_subplot(2, 1, 1)
    bars1 = ax1.bar(agent_labels, plot_costs, width=0.4, label=('1', '2', '3', '4', '5', '6'))
    ax1.errorbar(agent_labels, plot_costs, plot_costs_error, fmt="none", ecolor='r', capsize=4)
    ax1.set_ylabel('mean AoI')
    ax1.bar_label(bars1, label_type='center')

    ax2 = fig.add_subplot(2, 1, 2)
    bars2 = ax2.bar(agent_labels, plot_risky_states, width=0.4)
    ax2.errorbar(agent_labels, plot_risky_states, plot_risky_states_error, fmt='none', ecolor='r', capsize=4)
    ax2.set_ylabel('mean risky states')
    ax2.bar_label(bars2, label_type='center')

    fig.tight_layout()
    fig.savefig('results/plot/' + str(config_id) + '/' + file_name + '_sensitivityComparison.png')


def plot_bhistory(config_id, train_eps, test_eps, num_runs):
    """Plots the battery charge during testing.

    Parameters
    ----------
    config_id : float
        The config_id of the simulation data to plot.
    train_eps : int
        The number of episodes trained.
    test_eps : int
        The number of episodes tested.
    num_runs : int
        The number of runs simulated.

    Returns
    -------

    """

    _, _, _, p, measure, _, _, _, _, _, _, _, new_package_prob, _, _, _ = configs.load(config_id)

    folder_name = '_'.join([str(train_eps), str(test_eps), str(num_runs)])
    try:
        battery_history_TQL_complete = np.load('results/' + str(config_id) + '/' + folder_name + '/bh_TQL.npy')
        battery_history_QLF_complete = np.load('results/' + str(config_id) + '/' + folder_name + '/bh_QLF.npy')
        battery_history_threshold_complete = np.load('results/' + str(config_id) + '/' + folder_name + '/bh_threshold.npy')
        battery_history_random_complete = np.load('results/' + str(config_id) + '/' + folder_name + '/bh_random.npy')
    except OSError:
        print('No simulation data present for those parameters. Run the simulation and rerun this script.')
        raise OSError

    fig = plt.figure(dpi=100, figsize=(6.4, 6.4))

    ax0 = fig.add_subplot(4, 1, 1)
    ax0.set_title('TQL battery history')
    ax0.set_xlabel('episodes')
    ax0.set_ylabel('battery charge')
    for row in range(0, min(constants.No_battery_history, num_runs)):
        ax0.plot(range(0, constants.battery_history_length),
                 battery_history_TQL_complete[row][:constants.battery_history_length])

    ax1 = fig.add_subplot(4, 1, 2)
    ax1.set_title('QLF battery history')
    ax1.set_xlabel('episodes')
    ax1.set_ylabel('battery charge')
    for row in range(0, min(constants.No_battery_history, num_runs)):
        ax1.plot(range(0, constants.battery_history_length),
                 battery_history_QLF_complete[row][:constants.battery_history_length])

    ax2 = fig.add_subplot(4, 1, 3)
    ax2.set_title('threshold battery history')
    ax2.set_xlabel('episodes')
    ax2.set_ylabel('battery charge')
    for row in range(0, min(constants.No_battery_history, num_runs)):
        ax2.plot(range(0, constants.battery_history_length),
                 battery_history_threshold_complete[row][:constants.battery_history_length])

    ax3 = fig.add_subplot(4, 1, 4)
    ax3.set_title('random battery history')
    ax3.set_xlabel('episodes')
    ax3.set_ylabel('battery charge')
    for row in range(0, min(constants.No_battery_history, num_runs)):
        ax3.plot(range(0, constants.battery_history_length),
                 battery_history_random_complete[row][:constants.battery_history_length])

    fig.tight_layout()
    fig.savefig('results/plot/' + str(config_id) + '/' + folder_name + '_bhistory.png')


def plot_statecount(config_id, train_eps, test_eps, num_runs, traintest, agent):
    """Plot a chart which displays the ammount of times each state has been visited.

    Parameters
    ----------
    config_id : float
        The config in interest.
    train_eps : int
        The number of training episodes of the simulation in interest.
    test_eps : int
        The number of test episodes of the simulation in interest.
    num_runs : int
        The number of runs of the simulation in interest.
    traintest : str
        'train' to display the count during training, 'test' for the count during testing.
    agent : str
        The agent in interest: 'TQL', 'QLF', 'threshold', 'random'

    Returns
    -------

    """

    _, _, _, p, measure, _, _, _, _, _, _, _, new_package_prob, _, _, _ = configs.load(config_id)

    folder_name = '_'.join([str(train_eps), str(test_eps), str(num_runs)])
    try:
        statecount_complete = np.load(
            'results/' + str(config_id) + '/' + folder_name + '/sc_' + agent + '_' + traintest + '.npy')
    except OSError:
        print('No simulation data present for those parameters. Run the simulation and rerun this script.')
        raise OSError
    if traintest == 'train':
        num_eps = num_runs * train_eps
    elif traintest == 'test':
        num_eps = num_runs * test_eps
    else:
        raise

    statecount_complete_bsum = np.sum(np.sum(statecount_complete, axis=0), axis=2)
    statecount_complete_bsum_log = np.ma.log10(statecount_complete_bsum / num_eps)
    fig = plt.figure(dpi=100, figsize=(6.4, 4.8))

    x, y = np.meshgrid(np.arange(0, constants.aoi_cap + 1, 1), np.arange(0, constants.aoi_cap + 1, 1))
    ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    z_max = statecount_complete_bsum_log.max()
    z_min = statecount_complete_bsum_log.min()

    c = ax1.pcolormesh(x, y, statecount_complete_bsum_log, cmap='Blues', vmin=z_min, vmax=z_max)  # old cmap GnBu
    fig.colorbar(c, ax=ax1)
    ax1.set_ylabel('AoI sender')
    ax1.set_xlabel('AoI receiver')
    fig.savefig('results/plot/' + str(config_id) + '/' + folder_name + '_statecount_' + agent + '_' + traintest + '.png')


def plot_cost_history_training(config_id, train_eps, test_eps, num_runs, agent):
    """Plot the cost during training of the Q-values over the training episodes.

    Parameters
    ----------
    config_id : float
        The config_id of the simulation data to plot.
    train_eps : int
        The number of episodes trained.
    test_eps : int
        The number of episodes tested.
    num_runs : int
        The number of runs simulated.
    agent : str
        The agent in interest: 'TQL', 'QLF'

    Returns
    -------
    none

    """

    _, _, _, p, measure, _, _, _, _, _, _, _, new_package_prob, _, _, _ = configs.load(config_id)

    folder_name = '_'.join([str(train_eps), str(test_eps), str(num_runs)])
    try:
        pure_cost_history_training_complete = np.load('results/' + str(config_id) + '/' + folder_name + '/pure_cost_history_training_complete_' + agent +'.npy')
        cost_history_training_complete = np.load('results/' + str(config_id) + '/' + folder_name + '/cost_history_training_complete_' + agent +'.npy')
    except OSError:
        print('No simulation data present for those parameters. Run the simulation and rerun this script.')
        raise OSError
    fig = plt.figure(dpi=100, figsize=(6.4, 4.8))

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.set_title('tabular cost')
    ax1.set_xlabel('epsiodes')
    for row in range(0, min(constants.No_cost_history, num_runs)):
        ax1.plot(range(constants.ma_window, train_eps + 1), utils.moving_average(cost_history_training_complete[row],
                                                                             constants.ma_window))
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.set_title('tabular pure-cost')
    for row in range(0, min(constants.No_cost_history, num_runs)):
        ax2.plot(range(constants.ma_window, train_eps + 1), utils.moving_average(pure_cost_history_training_complete[row],
                                                                             constants.ma_window))
    fig.tight_layout()
    fig.savefig('results/plot/' + str(config_id) + '/' + folder_name + '_cost_history_training_' + agent +'.png')
