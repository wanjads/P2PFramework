import numpy as np

import constants
import parameters
import utils


# @abstract
class Agent:
    """abstract class for an agent

    This class models an abstract agent, which chooses an action based on an Markov Decision Process.


    Attributes
    ----------
    state_count_testing : ndarray
        An array to log how many times each state was visited during testing.

    Methods
    -------
    action_value
        should not be called for a general agent.
    test
        Tests the agent on a given environment.
    """

    def __init__(self):
        self.state_count_testing = np.zeros((constants.aoi_cap + 1, constants.aoi_cap + 1, parameters.B_max + 1))

    @staticmethod
    def action_value(obs):
        raise Exception('You constructed a general agent. The general agent does not have an action-value method '
                        'defined. Construct a specific agent (random/tabular/thershold) instead.')

    def test(self, environment):
        """Tests the agent on a given environment

        Parameters
        ----------
        environment : Environment
            The environment to test the agent on.
        render : bool
            to be removed

        Returns
        -------
        avg_cost : float
            The average cost over all timestamps during the test.
        avg_pure_cost : float
            The average pure_cost over all timesteps during the test.
        avg_risky_states : float
            The relative frequency of a state being a risky state during the test.
        battery_history : ndarray
            The history of the battery charge during the test.
        energy_running_mean : ndarray
            The mean energy spent per timestep during the test.
        """

        test_eps = parameters.test_eps
        print("----------   TEST STRATEGY   ----------") if parameters.print_mode == 1 else None

        # create arrays/variables to be filled with data/results during/after the test
        pure_costs = []
        costs = []
        risky_states = []
        battery_history = []
        energy_running_mean = 0

        # RESET environment
        obs, done, ep_cost = environment.reset(), False, 0
        # create dictionary for known observations for better speed
        known_obs = {}

        # TEST
        for episode_no in range(test_eps):
            # Conduct timestep
            if tuple(obs) in known_obs.keys():
                action = known_obs[tuple(obs)]
            else:
                action = self.action_value(obs[None, :])
                known_obs.update({tuple(obs): action})
            obs, cost, done, [pure_cost, risky_state] = environment.step(action)
            # store data
            pure_costs += [pure_cost]
            costs += [cost]
            risky_states += [risky_state]
            battery_history += [obs[2]]
            self.state_count_testing[obs[0], obs[1], obs[2]] += 1
            action_energy = \
                constants.action[action][0] * parameters.sense_energy + \
                constants.action[action][1] * parameters.send_energy
            energy_running_mean = utils.running_mean(episode_no, energy_running_mean, action_energy)

            # print (and render)
            if episode_no % int(0.1 * test_eps) == 0 and parameters.print_mode == 1:
                print(str(int(episode_no / test_eps * 100)) + " %")

        # calculate averages
        avg_pure_cost = sum(pure_costs) / len(pure_costs)
        avg_cost = sum(costs) / len(costs)
        avg_risky_states = sum(risky_states) / len(risky_states)

        if parameters.print_mode == 1:
            print("100 %")

            print("avg pure cost: " + str(avg_pure_cost))
            print("avg risky states: " + str(avg_risky_states))

            print("----------   TEST COMPLETE   ----------")
            print()

        if parameters.energy_harvesting in ("constrained", "harvested"):
            return avg_pure_cost, avg_risky_states, battery_history, energy_running_mean
        elif parameters.energy_harvesting == "unlimited":
            return avg_cost, avg_risky_states, battery_history, energy_running_mean
