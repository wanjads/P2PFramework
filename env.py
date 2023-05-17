import random
import constants
import numpy as np
import decimal

import parameters


class Environment:
    """Modelling of the P2P environment.

    Models the core P2P environment and additional customizable components.
    The system core consists of a sender-receiver pair, a noisy channel and a perfect feedback channel.
    Customizable components are a battery with different charging options
    and the underlying process which provides status updates.
    Three different measures are calculated during the simulation.
    For further description consult Section 2 in the referenced paper (readme).

    Attributes
    ----------
    send_prob : float
        The quality of the transmission channel as probability for successful transmission.
    episode : int
        The current timestep the environment is in.
    aoi_sender: int
        The Age of Information of the latest status update at the sender.
    aoi_receiver : int
        The Age of Information of the latest received status update at the receiver.
    process_state : int
        The id of the current process state of the underlying process.
    information_at_sender : int
        The id of the process state in which the underlying process was in,
        when the last status update was sensed (randomly or by action).
    information_at_receiver : int
        The information (a process state id) in the latest package, which was sent and decoded to/at the receiver.
    aoii : int
        The aoii of the model. For explanation of the measure and its calculation view II.B.4 in the referenced paper.
    measure : str
        The used measure. This also influences the used cost function.
    risk_sensitivity : bool
        Toggles the risk-sensitivity in the calculation of the cost function.
    energy_sensitivity : bool
        Toggles the energy-sensitivity in the calculation of the cost function.
    battery : int
        The battery charge of the sender. Gets (dis)charged depending on the chosen parameters and configuration.
    battery_hd : decimal
        A high definition version of the battery charge
        which allows charging with energy packages which are not a whole number.


    Methods
    -------
    reset
        Resets the environment.
    step
        Executes one timestep
    """

    def __init__(self, measure, risk_sensitivity, energy_sensitivity, p):
        self.send_prob = p
        self.episode = 0
        self.aoi_sender = 0
        self.aoi_receiver = 1

        # further measures
        self.process_state = 0
        self.information_at_sender = 0
        self.information_at_receiver = 0
        self.aoii = 0

        self.measure = measure
        self.risk_sensitivity = risk_sensitivity
        self.energy_sensitivity = energy_sensitivity

        self.battery = constants.battery_start
        self.battery_hd = decimal.Decimal(constants.battery_start)

    def reset(self):
        """Resets the environment.

        Resets the environment, so that a new test or training simulation can be run on it.

        Returns
        -------
         : ndarray
            The state of the environment after resetting it.
        """
        self.episode = 0
        self.aoi_sender = 0
        self.aoi_receiver = 1
        self.process_state = 0
        self.information_at_sender = 0
        self.information_at_receiver = 0
        self.aoii = 0

        self.battery = constants.battery_start
        self.battery_hd = decimal.Decimal(constants.battery_start)

        if self.measure in ("AoI", "QAoI"):
            return np.array((self.aoi_sender, self.aoi_receiver, self.battery))
        elif self.measure == "AoII":
            return np.array((int(self.information_at_sender == self.information_at_receiver),
                             self.aoii,
                             self.battery))

    def step(self, a):
        """Executes one timestep

        Based on the action the environment executes a single timestep.

        Parameters
        ----------
        a : int
            The action the environment processes.

        Returns
        -------
        new_state : ndarray
            The new state of the model after processing the action.
        c : float
            The cost of the processed action.
        done : bool
            Triggers a reset of the env during training of the q_values.

        pure_cost : float
            A different cost of the processed action consisting only of the used measure.
        risky_state : bool
            Indicates if the new state is a risky state. The definition of a risky state depends on the measure.
        """

        done = False

        # decode action
        sense, send = constants.action[a]
        if parameters.sensing == 'active':
            if sense * parameters.sense_energy + send * parameters.send_energy > self.battery:
                sense, send = 0, 0
        else:
            if send * parameters.send_energy > self.battery:
                send = 0

        # update sender
        # senses a new package by random or by action
        if parameters.sensing == 'active':
            if sense:
                self.aoi_sender = 0
                self.information_at_sender = self.process_state
        elif parameters.sensing == 'random':
            if random.random() < parameters.new_package_prob:
                self.aoi_sender = 0
                self.information_at_sender = self.process_state

        # transmission attempt
        if send:
            if random.random() < self.send_prob:
                self.aoi_receiver = self.aoi_sender
                self.information_at_receiver = self.information_at_sender
                self.aoii = self.aoi_sender

        # discharge battery
        if parameters.energy_harvesting == 'harvested':
            self.battery += -1 * send * parameters.send_energy - 1 * sense * parameters.sense_energy
        elif parameters.energy_harvesting == 'constrained':
            self.battery_hd += decimal.Decimal(-1 * send * parameters.send_energy - 1 * sense * parameters.sense_energy)
            self.battery = int(np.floor(self.battery_hd))

        # increase AoI_receiver and aoii
        if self.aoi_receiver + 1 > constants.aoi_cap or self.aoii + 1 > constants.aoi_cap:
            raise Exception('The AoI-cap has been reached. Increase the cap in constants.py and rerun the simulation.')
        else:
            self.aoi_receiver += 1
            self.aoii += 1

        # set aoii
        if self.process_state == self.information_at_receiver:
            self.aoii = 0

        # DIFFERENT COST FUNCTIONS INCLUDING RISK
        risky_state = 0
        c = 0
        # AoI
        if self.measure == "AoI":
            pure_cost = self.aoi_receiver
            if parameters.energy_harvesting in ('harvested', 'constrained'):
                if self.energy_sensitivity:
                    c = self.aoi_receiver + \
                        constants.beta(self.battery) * (parameters.send_energy * send + parameters.sense_energy * sense)
                else:
                    c = self.aoi_receiver
            elif parameters.energy_harvesting == 'unlimited':
                c = self.aoi_receiver + parameters.send_energy * send + parameters.sense_energy * sense
            risky_state = int(self.aoi_receiver >= 5)

        # AoII
        elif self.measure == "AoII":
            pure_cost = self.aoii
            if parameters.energy_harvesting in ('harvested', 'constrained'):
                c = constants.beta(self.battery) * parameters.send_energy * a + self.aoii
            elif parameters.energy_harvesting == 'unlimited':
                c = parameters.send_energy * a + self.aoii
            risky_state = int(self.aoii >= 3)

        # QAoI
        elif self.measure == "QAoI":
            query_time_step = (random.random() < parameters.query_probability)
            pure_cost = query_time_step * self.aoi_receiver
            if parameters.energy_harvesting in ('harvested', 'constrained'):
                c = constants.beta(self.battery) * (parameters.send_energy * send + parameters.sense_energy * sense) + \
                    query_time_step * self.aoi_receiver
            elif parameters.energy_harvesting == 'unlimited':
                c = query_time_step * self.aoi_receiver
            risky_state = (int(self.episode % parameters.query_probability == 0) * self.aoi_receiver >= 5)

        # risk_sensitivity
        # increase cost if in a risky state
        if self.risk_sensitivity and risky_state == 1:
            c = parameters.risk_factor * c

        # new episode
        self.episode += 1

        # new process state
        if random.random() >= parameters.p_remain:
            old_state = self.process_state
            while self.process_state == old_state:
                self.process_state = random.randint(0, parameters.N_process_states - 1)

        # increase AoI_sender
        if self.aoi_sender + 1 > constants.aoi_cap:
            raise Exception('The AoI-cap has been reached. Increase the cap in constants.py and rerun the simulation.')
        else:
            self.aoi_sender += 1

        # charge battery
        if parameters.energy_harvesting == 'harvested':
            self.battery = np.minimum(self.battery + int(np.around(random.random() * parameters.h_max)),
                                      parameters.B_max)
        if parameters.energy_harvesting == 'constrained':
            self.battery_hd = decimal.Decimal(np.minimum(self.battery_hd + parameters.budget, parameters.B_max))
            self.battery = int(np.floor(self.battery_hd))

        # construct new state
        if self.measure in ("AoI", "QAoI"):
            new_state = np.array((self.aoi_sender, self.aoi_receiver, self.battery))
        elif self.measure == "AoII":
            new_state = np.array((int(self.information_at_sender == self.information_at_receiver), self.aoii,
                                  self.battery))

        return new_state, c, done, [pure_cost, risky_state]
