import argparse
import logging
import os

import constants
import parameters
from env import Environment
import baselines
from tabular import TabularQAgent
import numpy as np
import multiprocessing
import decimal

# Set the accuracy for the decimal class.
decimal.getcontext().prec = 4

# parse basic arguments
parser = argparse.ArgumentParser()
parser.add_argument('-rn', '--num_runs', type=int, default=parameters.num_runs)
parser.add_argument('-lr', '--learning_rate', type=float, default=7e-3)  # 7e-3 for AoI, 7e-4 for AoII
parser.add_argument('-r', '--render_test', action='store_true', default=False)
parser.add_argument('-p', '--plot_results', action='store_true', default=False)


class Process(multiprocessing.Process):
    """Process handling a single simulation.

    Each process handles a specific simulation simulating a configuration with additional simulation parameters such as:
    num_runs, learning_rate, energy_sensitivity, prime_mode...

    Attributes
    ----------
    p : float
        The quality of the transmission channel as probability for successful transmission.
    measure : str
        The used measure.
    risk_sensitivity : bool
        Decides if the simulation should operate risk-sensitive or not.
    num_runs : int
        The number of times the simulation is sequentially executed to calculate reliable results as the mean.
    learning_rate : float
        The learning rate for the Q-learning algorithm.
    sim_only : bool
        Limits the process to just the simulation and turns off the creation of plots.

    Methods
    -------
    run
        Starts a simulation.
    """

    def __init__(self, send_prob, m, risk_sensitivity, num_runs, learning_rate, sim_only):
        """Constructor

        Standard constructor. For documentation of the parameters view the class documentation.
        """
        super(Process, self).__init__()
        self.p = send_prob
        self.measure = m
        self.risk_sensitivity = risk_sensitivity
        self.num_runs = num_runs
        self.learning_rate = learning_rate

        self.sim_only = sim_only

    def run(self):
        """Starts a simulation consisting of multiple runs."""

        # PRINT basic configuration info
        print('---------- START SIMULATION -----------')
        print("configuration    : " + str(parameters.config_id)) if parameters.print_mode == 1 else None
        print("measure          : " + self.measure) if parameters.print_mode == 1 else None
        print("sensing mechanism: " + parameters.sensing) if parameters.print_mode == 1 else None
        print("energy supply    : " + parameters.energy_harvesting) if parameters.print_mode == 1 else None
        print("risk sensitivity : " + str(self.risk_sensitivity)) if parameters.print_mode == 1 else None
        print('For further parameters view the config file.')

        # create arrays to be filled with data/results during/after the simulation
        pure_costs_TQL = []  # means pure cost, without the battery and sending cost, just AoI cost
        risky_states_TQL = []
        pure_costs_QLF = []  # means pure cost, without the battery and sending cost, just AoI cost
        risky_states_QLF = []
        pure_costs_threshold = []
        risky_states_threshold = []
        pure_costs_random = []
        risky_states_random = []

        battery_history_TQL_complete = np.zeros((self.num_runs, parameters.test_eps))
        battery_history_QLF_complete = np.zeros((self.num_runs, parameters.test_eps))
        battery_history_threshold_complete = np.zeros((self.num_runs, parameters.test_eps))
        battery_history_random_complete = np.zeros((self.num_runs, parameters.test_eps))

        state_count_TQL_training_complete = np.zeros(
            (self.num_runs, constants.aoi_cap + 1, constants.aoi_cap + 1, parameters.B_max + 1))
        state_count_TQL_testing_complete = np.zeros(
            (self.num_runs, constants.aoi_cap + 1, constants.aoi_cap + 1, parameters.B_max + 1))
        state_count_QLF_training_complete = np.zeros(
            (self.num_runs, constants.aoi_cap + 1, constants.aoi_cap + 1, parameters.B_max + 1))
        state_count_QLF_testing_complete = np.zeros(
            (self.num_runs, constants.aoi_cap + 1, constants.aoi_cap + 1, parameters.B_max + 1))
        state_count_threshold_testing_complete = np.zeros(
            (self.num_runs, constants.aoi_cap + 1, constants.aoi_cap + 1, parameters.B_max + 1))
        state_count_random_testing_complete = np.zeros(
            (self.num_runs, constants.aoi_cap + 1, constants.aoi_cap + 1, parameters.B_max + 1))
        cost_history_training_complete_TQL = np.zeros(
            (self.num_runs, parameters.train_eps))
        pure_cost_history_training_complete_TQL = np.zeros(
            (self.num_runs, parameters.train_eps))
        cost_history_training_complete_QLF = np.zeros(
            (self.num_runs, parameters.train_eps))
        pure_cost_history_training_complete_QLF = np.zeros(
            (self.num_runs, parameters.train_eps))

        energy_mean_TQL_complete = []
        energy_mean_QLF_complete = []
        energy_mean_threshold_complete = []
        energy_mean_random_complete = []

        for run in range(self.num_runs):
            print("run: " + str(run)) if parameters.print_mode == 1 or parameters.print_mode == 2 else None

            # CREATION of environments and agents
            # env gets constructed with both sensitivities set to false. They get changed anyway.
            env = Environment(self.measure, False, False, self.p)

            env.risk_sensitivity, env.energy_sensitivity = False, False
            TQL_agent = TabularQAgent(self.measure, self.p, False, lr=self.learning_rate)
            TQL_agent.train(env)

            env.risk_sensitivity, env.energy_sensitivity = parameters.risk_sensitivity, True
            QLF_agent = TabularQAgent(self.measure, self.p, True, lr=self.learning_rate)
            QLF_agent.train(env)

            threshold_agent = baselines.ThresholdAgent(self.measure, parameters.threshold)
            random_agent = baselines.RandomAgent()

            # TESTING
            run_pure_costs_TQL, run_risky_states_TQL, battery_history_TQL, energy_mean_TQL = TQL_agent.test(env)
            run_pure_costs_QLF, run_risky_states_QLF, battery_history_QLF, energy_mean_QLF = QLF_agent.test(env)
            run_pure_costs_threshold, run_risky_states_threshold, battery_history_threshold, energy_mean_threshold = \
                threshold_agent.test(env)
            run_pure_costs_random, run_risky_states_random, battery_history_random, energy_mean_random = \
                random_agent.test(env)

            pure_costs_TQL += [run_pure_costs_TQL]
            risky_states_TQL += [run_risky_states_TQL]
            pure_costs_QLF += [run_pure_costs_QLF]
            risky_states_QLF += [run_risky_states_QLF]
            pure_costs_threshold += [run_pure_costs_threshold]
            risky_states_threshold += [run_risky_states_threshold]
            pure_costs_random += [run_pure_costs_random]
            risky_states_random += [run_risky_states_random]

            battery_history_TQL_complete[run] = battery_history_TQL
            battery_history_QLF_complete[run] = battery_history_QLF
            battery_history_threshold_complete[run] = battery_history_threshold
            battery_history_random_complete[run] = battery_history_random

            state_count_TQL_training_complete[run] = TQL_agent.state_count_training
            state_count_QLF_testing_complete[run] = QLF_agent.state_count_testing
            state_count_threshold_testing_complete[run] = threshold_agent.state_count_testing
            state_count_random_testing_complete[run] = random_agent.state_count_testing

            pure_cost_history_training_complete_TQL[run] = TQL_agent.pure_cost_history_training
            cost_history_training_complete_TQL[run] = TQL_agent.cost_history_training
            pure_cost_history_training_complete_QLF[run] = QLF_agent.pure_cost_history_training
            cost_history_training_complete_QLF[run] = QLF_agent.cost_history_training

            energy_mean_TQL_complete += [energy_mean_TQL]
            energy_mean_QLF_complete += [energy_mean_QLF]
            energy_mean_threshold_complete += [energy_mean_threshold]
            energy_mean_random_complete += [energy_mean_random]

        # average all runs
        results = np.array([np.mean(pure_costs_TQL), np.mean(pure_costs_QLF), np.mean(pure_costs_threshold),
                            np.mean(pure_costs_random),
                            np.mean(risky_states_TQL), np.mean(risky_states_QLF), np.mean(risky_states_threshold),
                            np.mean(risky_states_random),
                            np.std(pure_costs_TQL), np.std(pure_costs_QLF), np.std(pure_costs_threshold),
                            np.std(pure_costs_random),
                            np.std(risky_states_TQL), np.std(risky_states_QLF), np.std(risky_states_threshold),
                            np.std(risky_states_random)])

        # SAVE results/data
        folder_name = '_'.join([str(parameters.train_eps),
                                str(parameters.test_eps), str(self.num_runs)])
        try:
            os.mkdir('results/' + str(parameters.config_id) + '/' + folder_name)
        except FileExistsError:
            print('folder already present')

        np.save('results/' + str(parameters.config_id) + '/' + folder_name + '/results.npy', results)
        np.save('results/' + str(parameters.config_id) + '/' + folder_name + '/bh_TQL.npy',
                battery_history_TQL_complete)
        np.save('results/' + str(parameters.config_id) + '/' + folder_name + '/bh_QLF.npy',
                battery_history_QLF_complete)
        np.save('results/' + str(parameters.config_id) + '/' + folder_name + '/bh_threshold.npy',
                battery_history_threshold_complete)
        np.save('results/' + str(parameters.config_id) + '/' + folder_name + '/bh_random.npy',
                battery_history_random_complete)
        np.save('results/' + str(parameters.config_id) + '/' + folder_name + '/sc_TQL_train.npy',
                state_count_TQL_training_complete)
        np.save('results/' + str(parameters.config_id) + '/' + folder_name + '/sc_TQL_test.npy',
                state_count_TQL_testing_complete)
        np.save('results/' + str(parameters.config_id) + '/' + folder_name + '/sc_QLF_train.npy',
                state_count_QLF_training_complete)
        np.save('results/' + str(parameters.config_id) + '/' + folder_name + '/sc_QLF_test.npy',
                state_count_QLF_testing_complete)
        np.save('results/' + str(parameters.config_id) + '/' + folder_name + '/sc_threshold_test.npy',
                state_count_threshold_testing_complete)
        np.save('results/' + str(parameters.config_id) + '/' + folder_name + '/sc_random_test.npy',
                state_count_random_testing_complete)
        TQL_agent.save_qvalues('results/' + str(parameters.config_id) + '/' + folder_name, 'TQL')
        QLF_agent.save_qvalues('results/' + str(parameters.config_id) + '/' + folder_name, 'QLF')
        np.save('results/' + str(parameters.config_id) + '/' + folder_name + '/cost_history_training_complete_TQL.npy',
                cost_history_training_complete_TQL)
        np.save(
            'results/' + str(parameters.config_id) + '/' + folder_name + '/pure_cost_history_training_complete_TQL.npy',
            pure_cost_history_training_complete_TQL)
        np.save('results/' + str(parameters.config_id) + '/' + folder_name + '/cost_history_training_complete_QLF.npy',
                cost_history_training_complete_QLF)
        np.save(
            'results/' + str(parameters.config_id) + '/' + folder_name + '/pure_cost_history_training_complete_QLF.npy',
            pure_cost_history_training_complete_QLF)

        # PRINT results
        print(str(parameters.config_id) + " " + "mean_pure_cost_TQL: " + str(np.mean(pure_costs_TQL)))
        print(str(parameters.config_id) + " " + "mean_risky_states_TQL: " + str(np.mean(risky_states_TQL)))
        print(str(parameters.config_id) + " " + "mean_pure_cost_QLF: " + str(np.mean(pure_costs_QLF)))
        print(str(parameters.config_id) + " " + "mean_risky_states_QLF: " + str(np.mean(risky_states_QLF)))
        print(str(parameters.config_id) + " " + "mean_pure_cost_threshold: " + str(np.mean(pure_costs_threshold)))
        print(str(parameters.config_id) + " " + "mean_risky_states_threshold: " + str(np.mean(risky_states_threshold)))
        print(str(parameters.config_id) + " " + "mean_pure_cost_random: " + str(np.mean(pure_costs_random)))
        print(str(parameters.config_id) + " " + "mean_risky_states_random: " + str(np.mean(risky_states_random)))

        print(str(parameters.config_id) + " " + "stddev_pure_cost_TQL: " + str(np.std(pure_costs_TQL)))
        print(str(parameters.config_id) + " " + "stddev_risky_states_TQL: " + str(np.std(risky_states_TQL)))
        print(str(parameters.config_id) + " " + "stddev_pure_cost_QLF: " + str(np.std(pure_costs_QLF)))
        print(str(parameters.config_id) + " " + "stddev_risky_states_QLF: " + str(np.std(risky_states_QLF)))
        print(str(parameters.config_id) + " " + "stddev_pure_cost_threshold: " + str(np.std(pure_costs_threshold)))
        print(str(parameters.config_id) + " " + "stddev_risky_states_threshold: " + str(np.std(risky_states_threshold)))
        print(str(parameters.config_id) + " " + "stddev_pure_cost_random: " + str(np.std(pure_costs_random)))
        print(str(parameters.config_id) + " " + "stddev_risky_states_random: " + str(np.std(risky_states_random)))

        print(str(parameters.config_id) + " " + "energy_mean_TQL: " + str(np.mean(energy_mean_TQL_complete)))
        print(str(parameters.config_id) + " " + "energy_mean_QLF: " + str(np.mean(energy_mean_QLF_complete)))
        print(
            str(parameters.config_id) + " " + "energy_mean_threshold: " + str(np.mean(energy_mean_threshold_complete)))
        print(str(parameters.config_id) + " " + "energy_mean_random: " + str(np.mean(energy_mean_random_complete)))
        print(str(parameters.config_id) + " " + "energy_stddev_TQL: " + str(np.std(energy_mean_TQL_complete)))
        print(str(parameters.config_id) + " " + "energy_stddev_QLF: " + str(np.std(energy_mean_QLF_complete)))
        print(
            str(parameters.config_id) + " " + "energy_stddev_threshold: " + str(np.std(energy_mean_threshold_complete)))
        print(str(parameters.config_id) + " " + "energy_stddev_random: " + str(np.std(energy_mean_random_complete)))

        #  Plotting
        if not self.sim_only:
            pass


if __name__ == '__main__':

    # ARGUMENT PARSER
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.INFO)

    # CREATE FOLDER STRUCTURE FOR RESULTS
    try:
        print('Create folder structure for saving the results. (Step 1)')
        os.mkdir('results/plot/')
    except FileExistsError:
        print('Folder structure is already present.')
    try:
        print('Create folder structure for saving the results. (Step 2)')
        os.mkdir('results/' + str(parameters.config_id) + '/')
        os.mkdir('results/plot/' + str(parameters.config_id) + '/')
    except FileExistsError:
        print('Folder structure is already present.')

    # Spawn and run the process
    p = Process(parameters.channel_quality, parameters.measure, parameters.risk_sensitivity, args.num_runs,
                args.learning_rate, parameters.sim_only)
    p.run()
