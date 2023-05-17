import constants
import parameters
from agent import Agent
import numpy as np


class RandomAgent(Agent):
    """This agent acts randomly.

    This agent acts randomly. The frequency of an action (vs. idle) is matched to the parameters of the config.

    Attributes
    ----------
    state_count_testing : ndarray
        An array to log how many times each state was visited during testing.

    Methods
    -------
    action_value
        Randomly chooses an action value.
    test
        Tests the agent on a given environment.
    """

    def __init__(self):
        super().__init__()

    def action_value(self, obs):
        """Randomly chooses an action value.

        Parameters
        ----------
        obs : ndarray

        Returns
        -------
        a : int
            The chosen action.
        """
        if obs[0][2] < parameters.send_energy + parameters.sense_energy:
            return 0
        else:
            if np.random.random() < constants.p_random:
                return 3
                # 3 is also valid for configs with an action space [0, 1],
                # because it gets decoded correctly into sending = 1.
            else:
                return 0


class ThresholdAgent(Agent):
    """This agent implements threshold-based strategies for different measures.

       The strategy is different for each config.

       Attributes
       ----------
       state_count_testing : ndarray
           An array to log how many times each state was visited during testing.
       measure : str
           The used measure.
       threshold : int
           The threshold used in a simple rule to decide if action should be taken.

       Methods
       -------
       action_value
           Chooses an action value.
       test
           Tests the agent on a given environment.
    """

    def __init__(self, measure, threshold):
        super().__init__()
        self.measure = measure
        self.threshold = threshold

    def action_value(self, obs):
        if self.measure == "AoI":
            if parameters.sensing == 'random':
                return int(obs[0][1] - obs[0][0] >= self.threshold)
            elif parameters.sensing == 'active':
                return 3 * int(obs[0][1] >= self.threshold)
        if self.measure == "AoII":
            return int(obs[0][1] >= self.threshold)
            # return int(obs[0][2] != obs[0][3]), None
        elif self.measure == "QAoI":
            if parameters.sensing == 'random':
                return int(obs[0][1] - obs[0][0] >= self.threshold)
            elif parameters.sensing == 'active':
                return 3 * int(obs[0][1] >= self.threshold)
