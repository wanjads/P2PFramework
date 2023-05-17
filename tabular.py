import random
import constants
import numpy as np

import parameters
from agent import Agent
import baselines
import copy


class TabularQAgent(Agent):
    """This agent implements tabular Q-learning.

    This agent implements tabular Q-learning and offers an option to prime the Q-values with a baseline agent.

    Attributes
    ----------
    measure : str
        The used measure.
    learning_rate : float
        The learning_rate balances the adjustment of the Q-values during training.
    aoi_cap : int
        The max value for the aoi.
    gamma : float
        The so called discount-factor, which balances how much the expected cost of the new state influences
        the adjustment of the Q-values during training.
    epsilon : float
        Parameter for the epsilon-greedy functionality,
        which decides if a random action is chosen to explore the state-space
        or if the learned action is chosen to exploit.
    delta : float
        Parameter for the epsilon-greedy functionality,
        which lowers the value of epsilon over time.
    p : float
        The quality of the transmission channel as probability for successful transmission.
    prime_mode : bool
        Toggles the priming of the Q-values before training.
    qvalues : ndarray
        Array which stores the Q-values.
    state_count_training : ndarray
        An array to log how many times each state was visited during testing.
    cost_history_training : ndarray
        The history of the cost during training.
    pure_cost_history_training : ndarray
        The history of the pure_cost during training.
    state_count_testing : ndarray
        An array to log how many times each state was visited during testing.

    Methods
    -------
    action_value
        Chooses an action based on the current Q-values.
    update
        Updates the Q-Values.
    train
        Trains the model.
    save_q_values
        Save the current Q-values.
    load_q_values
        Load Q-Values.
    prime_q_values
        Prime the Q-Values.
    """

    def __init__(self, measure, send_prob, prime_mode, lr=0.01, aoi_cap=constants.aoi_cap, gamma=0.7,
                 epsilon0=constants.epsilon_init, delta=constants.delta):
        super().__init__()

        self.measure = measure

        self.learning_rate = lr
        self.aoi_cap = aoi_cap
        self.gamma = gamma
        self.epsilon = epsilon0
        self.delta = delta
        self.p = send_prob

        self.prime_mode = prime_mode

        # initialise the Q-values depending on the size of the action-space
        if parameters.sensing == 'active':
            self.qvalues = np.zeros((aoi_cap + 1, aoi_cap + 1, parameters.B_max + 1, 4))
        elif parameters.sensing == 'random':
            self.qvalues = np.zeros((aoi_cap + 1, aoi_cap + 1, parameters.B_max + 1, 2))
        # prime Q-values
        if self.prime_mode:
            self.prime_q_values()

        self.state_count_training = np.zeros((aoi_cap + 1, aoi_cap + 1, parameters.B_max + 1))
        self.cost_history_training = np.zeros(parameters.train_eps)
        self.pure_cost_history_training = np.zeros(parameters.train_eps)

    def action_value(self, obs):
        """Chooses an action based on the current Q-values.

        Parameters
        ----------
        obs : ndarray
            The observation used to choose an action.

        Returns
        -------
        a : int
            The chosen action.
        """

        obs = obs[0]
        # check if energy is enough for any action
        if obs[2] < parameters.send_energy and obs[2] < parameters.sense_energy:
            return 0

        else:
            if parameters.sensing == 'active':
                # random action because of epsilon greedy
                if random.random() < self.epsilon:
                    action = random.randint(0, 3)
                # random action if Q-values are identical
                elif self.qvalues[obs[0]][obs[1]][obs[2]][0] == self.qvalues[obs[0]][obs[1]][obs[2]][1] == \
                        self.qvalues[obs[0]][obs[1]][obs[2]][2] == self.qvalues[obs[0]][obs[1]][obs[2]][3]:
                    action = random.randint(0, 3)
                # choose smallest Q-value
                else:
                    action = np.argmin(self.qvalues[obs[0]][obs[1]][obs[2]])
            elif parameters.sensing == 'random':
                # random action because of epsilon greedy
                if random.random() < self.epsilon:
                    action = random.randint(0, 1)
                # random action if Q-values are identical
                elif self.qvalues[obs[0]][obs[1]][obs[2]][0] == self.qvalues[obs[0]][obs[1]][obs[2]][1]:
                    action = random.randint(0, 1)
                # choose smallest Q-value
                else:
                    action = np.argmin(self.qvalues[obs[0]][obs[1]][obs[2]])

            # check if energy is sufficient for chosen action
            sense, send = constants.action[action]
            if (sense * parameters.sense_energy + send * parameters.send_energy) <= obs[2]:
                pass
            else:
                return 0
            return action

    def update(self, old_obs, obs, cost, action):
        """Update the Q-values based on the experienced gained from a timestep.

        Parameters
        ----------
        old_obs : ndarray
            The state of the environment before the action.
        obs : ndarray
            The state of the environment after the action.
        cost : float
            The cost of the action.
        action : int
            The processed action.

        Returns
        -------
        none

        """
        self.epsilon = self.delta * self.epsilon

        V = np.min(self.qvalues[obs[0]][obs[1]][obs[2]])
        old_q_value = self.qvalues[old_obs[0]][old_obs[1]][old_obs[2]][action]

        self.qvalues[old_obs[0]][old_obs[1]][old_obs[2]][action] = \
            (1 - self.learning_rate) * old_q_value + self.learning_rate * (cost + self.gamma * V)

    def train(self, environment):
        """Train the Q-values.

        Parameters
        ----------
        environment : Environment
            The environment to train the Q-values in.

        Returns
        -------
        none

        """
        train_eps = parameters.train_eps
        print("----------    TRAIN MODEL    ----------") if parameters.print_mode == 1 else None
        next_obs = environment.reset()
        for episode in range(train_eps):
            # save current state
            obs = copy.deepcopy(next_obs)
            # conduct timestep
            action = self.action_value(obs[None, :])
            next_obs, cost, done, [pure_cost, _] = environment.step(action)

            if done:
                next_obs = environment.reset()
            # update the Q-values
            self.update(obs, next_obs, cost, action)

            # log data
            self.state_count_training[next_obs[0], next_obs[1], next_obs[2]] += 1
            self.cost_history_training[episode] = cost
            self.pure_cost_history_training[episode] = pure_cost

            # print progress
            if episode % int(0.1 * train_eps) == 0 and parameters.print_mode == 1:
                print(str(int(episode / train_eps * 100)) + " %, epsilon:" + str(self.epsilon))
        if parameters.print_mode == 1:
            print("100 %")
            print("---------- TRAINING COMPLETE ----------")

    def save_qvalues(self, path, agent_name):
        """Save the current Q-values into a .npy file.

        Parameters
        ----------
        path : str
            The path to the location to save the file.
        agent_name : str
            The name (TQL, QLF) of the agent, which created the Q-values.

        Returns
        -------
        none

        """
        np.save(path + "/qvalues" + agent_name + ".npy", self.qvalues)

    def load_q_values(self, path, agent_name):
        """Load the Q-values from a .npy file (saved by the method save_qvalues).

        Returns
        -------
        none

        """
        self.qvalues = np.load(path + "/qvalues" + agent_name + ".npy")

    def prime_q_values(self):
        """Prime the Q-values.

        Primes the Q-values based on a threshold/baseline agent.

        Returns
        -------
        none

        """
        # initialise helper objects
        helper_threshold_agent = baselines.ThresholdAgent(self.measure, parameters.threshold)
        state_array = np.zeros((constants.aoi_cap + 1, constants.aoi_cap + 1, parameters.B_max + 1))

        # set all Q-values to the high priming value
        if parameters.sensing == 'active':
            self.qvalues = np.ones(
                (self.aoi_cap + 1, self.aoi_cap + 1, constants.parameters.B_max + 1, 4)) * constants.prime_value_high
            # self.qvalues[:, :, :, 1:3] = np.ones(
            #     (self.aoi_cap + 1, self.aoi_cap + 1, constants.battery_max + 1, 2)) * constants.prime_value_high / 2
        elif parameters.sensing == 'random':
            self.qvalues = np.ones(
                (self.aoi_cap + 1, self.aoi_cap + 1, parameters.B_max + 1, 2)) * constants.prime_value_high

        # loop over all states, choose action, set corresponding Q-value to low.
        for state, n in np.ndenumerate(state_array):
            action = helper_threshold_agent.action_value(
                np.reshape(state,
                           constants.obs_shape))
            q_value_position = np.concatenate((np.array(state), np.array([action])))
            self.qvalues[tuple(q_value_position)] = constants.prime_value_low
        print('q_Values primed.') if parameters.print_mode == 1 else None
