import parameters
from env import Environment
import baselines
import numpy as np


def run(threshold, runs):
    """Tests the threshold strategy for a given threshold.

    Parameters
    ----------
    threshold : int
        The threshold used for the test.
    runs : int
        The number of tests/runs conducted to calculate an accurate mean result.

    Returns
    -------
     : float
        The mean cost over all timesteps over all runs.

    """

    energy_sensitivity = False
    # irrelevant because the performance is judged on the pure cost

    # create arrays to be filled with data/results during/after the test
    costs_baseline = []
    risky_states_baseline = []

    energy_mean_baseline_complete = []

    for r in range(runs):

        # CREATION of environment and agent
        env = Environment(parameters.measure, parameters.risk_sensitivity, energy_sensitivity,
                          parameters.channel_quality)
        baseline_agent = baselines.ThresholdAgent(parameters.measure, threshold)

        # TESTING
        run_costs_baseline, run_risky_states_baseline, _, energy_mean_baseline \
            = baseline_agent.test(env)

        # STORE the data
        costs_baseline += [run_costs_baseline]
        risky_states_baseline += [run_risky_states_baseline]

        energy_mean_baseline_complete += [energy_mean_baseline]

    # average all runs
    result = np.array([np.mean(costs_baseline), np.mean(risky_states_baseline),
                       np.std(costs_baseline), np.std(risky_states_baseline)])

    # return only the mean cost
    return result[0]


def find(runs):
    """Find the best threshold for a given configuration.

    Parameters
    ----------
    runs : int
        The number of runs/tests to conduct and average over.

    Returns
    -------
     : int
        The best threshold
    """
    # Concept: see code
    t = 0
    if parameters.sensing == "active":
        t = 1
    results = []
    while True:
        result = run(t, runs)
        results += [result]
        print(result)
        if len(results) > 1 and results[-2] < results[-1]:
            print()
            print("Best threshold: " + str(t - 1))
            break
        t += 1

    return t - 1


if __name__ == '__main__':

    find(10)
