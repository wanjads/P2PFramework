import decimal


def load(config_id):
    """Loads specific variable values based on a config-id.

    Loads specific variable values, so-called parameters, based on a config id.
    The variable values control the specific type of simulation.
    The value of every variable is checked for they are validity.
    Read the readme.txt for information on how to create new configs.

    Parameters
    ----------
    config_id : float
        The config id, representing the configuration of the simulation.

    Returns
    -------
    N_process_states : int
        The number of process states. Relevant for measure = 'AoII'.
    sensing : str
        The sensing mode.
    energy_harvesting : str
        The energy harvesting mode.
    channel_quality : float
        The quality of the transmission channel as probability for successful transmission.
    measure : str
        The used measure.
    risk_sensitivity : bool
        Decides if the simulation should operate risk-sensitive or not.
    B_max : int
        The size of the battery in energy units.
    h_max : int
        The maximum amount of energy harvested each timestep i.e. the upper bound of the uniform distribution.
    p_remain : float
        The probability to remain in the current process state. Relevant for measure = 'AoII'.
    send_energy : int
        The amount of energy required for a transmission attempt.
        Also serves as a factor in the cost function in different ways.
    sense_energy : int
        The amount of energy required to sense a new status update at the sender.
        Also serves as a factor in the cost function in different ways.
    query_probability : float
        The probability for a timestep to be a query-timestep. Relevant for measure = 'QAoI'
    new_package_prob : float
        The probability to receive a new status update at the sender. Relevant for sensing = 'random'
    budget : float
        The energy budget per timestep which may not be exceeded. Relevant for energy_harvesting = 'constrained'.
    threshold : int
        The threshold for different threshold-based strategies.
        The exact function is dependent on the measure and on the sensing mode.
    """

    # Depending on the specific configuration, some values might be unnecessary. Insert valid placeholders.
    if config_id == 0.0:

        N_process_states = 10   # greater than 0, integer.

        sensing = 'active'  # 'random', 'active'
        energy_harvesting = 'unlimited'  # 'unlimited', 'constrained', 'harvested'
        channel_quality = 0.9  # 0.1, 0.2, ..., 1
        measure = 'AoI'  # 'AoI', 'AoII', 'QAoI'
        risk_sensitivity = True
        risk_factor = 2
        threshold = 3

        # optional parameters
        # energy_harvesting
        send_energy = 1
        sense_energy = 4
        # 'harvested'
        B_max = 5
        h_max = 1
        # 'constrained'
        budget = decimal.Decimal(0.8)

        # sensing
        # 'random'
        new_package_prob = 0.5

        # measure
        # 'AoII'
        p_remain = 0.5
        # 'QAoI'
        query_probability = 0.2

    elif config_id == 0.1:
        N_process_states = 10  # greater than 0, integer.
        sensing = 'random'  # 'random', 'active'
        energy_harvesting = 'unlimited'  # 'unlimited', 'constrained', 'harvested'
        channel_quality = 0.9  # 0.1, 0.2, ..., 1
        measure = 'AoI'  # 'AoI', 'AoII', 'QAoI'
        risk_sensitivity = True
        risk_factor = 2
        threshold = 0

        # optional parameters
        # energy_harvesting
        # special case: necessary energy for actions if 'harvested'
        # also weights in the cost function, also for 'unlimited'
        send_energy = 1
        sense_energy = 4
        # 'harvested'
        B_max = 5
        h_max = 1
        # 'constrained'
        budget = decimal.Decimal(1.5)

        # sensing
        # 'random'
        new_package_prob = 0.5

        # measure
        # 'AoII'
        p_remain = 0.5
        # 'QAoI'
        query_probability = 0.2

    elif config_id == 0.2:
        N_process_states = 10  # greater than 0, integer.
        sensing = 'random'  # 'random', 'active'
        energy_harvesting = 'unlimited'  # 'unlimited', 'constrained', 'harvested'
        channel_quality = 0.9  # 0.1, 0.2, ..., 1
        measure = 'AoI'  # 'AoI', 'AoII', 'QAoI'
        risk_sensitivity = True
        risk_factor = 2
        threshold = 1

        # optional parameters
        # energy_harvesting
        # special case: necessary energy for actions if 'harvested'
        # also weights in the cost function, also for 'unlimited'
        send_energy = 1
        sense_energy = 4
        # 'harvested'
        B_max = 5
        h_max = 1
        # 'constrained'
        budget = decimal.Decimal(0.8)

        # sensing
        # 'random'
        new_package_prob = 1.0

        # measure
        # 'AoII'
        p_remain = 0.5
        # 'QAoI'
        query_probability = 0.2

        threshold = 0

    elif config_id == 1.0:
        N_process_states = 10
        sensing = 'random'  # 'random', 'active'
        energy_harvesting = 'constrained'  # 'unlimited', 'constrained', 'harvested'
        channel_quality = 0.9  # 0.1, 0.2, ..., 1
        measure = 'AoII'  # 'AoI', 'AoII', 'QAoI'
        risk_sensitivity = True
        risk_factor = 2
        threshold = 1

        # optional parameters
        # energy_harvesting
        # special case: necessary energy for actions if 'harvested'
        # also weights in the cost function, also for 'unlimited'
        send_energy = 1
        sense_energy = 2
        # 'harvested'
        B_max = 10
        h_max = 1
        # 'constrained'
        budget = decimal.Decimal(0.5)

        # sensing
        # 'random'
        new_package_prob = 1.0

        # measure
        # 'AoII'
        p_remain = 0.5
        # 'QAoI'
        query_probability = 0.2

    elif config_id == 1.1:
        N_process_states = 10   # greater than 0, integer
        sensing = 'random'  # 'random', 'active'
        energy_harvesting = 'unlimited'  # 'unlimited', 'constrained', 'harvested'
        channel_quality = 0.9  # 0.1, 0.2, ..., 1
        measure = 'AoII'  # 'AoI', 'AoII', 'QAoI'
        risk_sensitivity = True
        risk_factor = 2
        threshold = 1

        # optional parameters
        # energy_harvesting
        # special case: necessary energy for actions if 'harvested'
        # also weights in the cost function, also for 'unlimited'
        send_energy = 1
        sense_energy = 2
        # 'harvested'
        B_max = 10
        h_max = 1
        # 'constrained'
        budget = decimal.Decimal(0.5)

        # sensing
        # 'random'
        new_package_prob = 1.0

        # measure
        # 'AoII'
        p_remain = 0.5
        # 'QAoI'
        query_probability = 0.2

    elif config_id == 1.2:
        N_process_states = 10   # greater than 0, integer
        sensing = 'random'  # 'random', 'active'
        energy_harvesting = 'harvested'  # 'unlimited', 'constrained', 'harvested'
        channel_quality = 0.9  # 0.1, 0.2, ..., 1
        measure = 'AoII'  # 'AoI', 'AoII', 'QAoI'
        risk_sensitivity = True
        risk_factor = 2
        threshold = 1

        # optional parameters
        # energy_harvesting
        # special case: necessary energy for actions if 'harvested'
        # also weights in the cost function, also for 'unlimited'
        send_energy = 1
        sense_energy = 2
        # 'harvested'
        B_max = 10
        h_max = 1
        # 'constrained'
        budget = decimal.Decimal(0.5)

        # sensing
        # 'random'
        new_package_prob = 1.0

        # measure
        # 'AoII'
        p_remain = 0.5
        # 'QAoI'
        query_probability = 0.2

    elif config_id == 2.0:
        N_process_states = 10   # greater than 0, integer
        sensing = 'random'  # 'random', 'active'
        energy_harvesting = 'harvested'  # 'unlimited', 'constrained', 'harvested'
        channel_quality = 0.9  # between 0 and 1 exclusive
        measure = 'QAoI'  # 'AoI', 'AoII', 'QAoI'
        risk_sensitivity = False
        risk_factor = 1
        threshold = 1

        # optional parameters
        # energy_harvesting
        # special case: necessary energy for actions if 'harvested'
        # also weights in the cost function, also for 'unlimited'
        send_energy = 1
        sense_energy = 0
        # 'harvested'
        B_max = 5
        h_max = 1
        # 'constrained'
        budget = decimal.Decimal(1.5)

        # sensing
        # 'random'
        new_package_prob = 0.8

        # measure
        # 'AoII'
        p_remain = 0.5
        # 'QAoI'
        query_probability = 1.0

        threshold = 1

    elif config_id == 2.1:
        N_process_states = 10   # greater than 0, integer
        sensing = 'random'  # 'random', 'active'
        energy_harvesting = 'harvested'  # 'unlimited', 'constrained', 'harvested'
        channel_quality = 0.9  # 0.1, 0.2, ..., 1
        measure = 'QAoI'  # 'AoI', 'AoII', 'QAoI'
        risk_sensitivity = False
        risk_factor = 1
        threshold = 1

        # optional parameters
        # energy_harvesting
        # special case: necessary energy for actions if 'harvested'
        # also weights in the cost function, also for 'unlimited'
        send_energy = 1
        sense_energy = 0
        # 'harvested'
        B_max = 5
        h_max = 1
        # 'constrained'
        budget = decimal.Decimal(0.2)

        # sensing
        # 'random'
        new_package_prob = 0.8

        # measure
        # 'AoII'
        p_remain = 0.5
        # 'QAoI'
        query_probability = 0.75

    elif config_id == 2.2:
        N_process_states = 10   # greater than 0, integer
        sensing = 'random'  # 'random', 'active'
        energy_harvesting = 'harvested'  # 'unlimited', 'constrained', 'harvested'
        channel_quality = 0.9  # 0.1, 0.2, ..., 1
        measure = 'QAoI'  # 'AoI', 'AoII', 'QAoI'
        risk_sensitivity = False
        risk_factor = 1
        threshold = 1

        # optional parameters
        # energy_harvesting
        # special case: necessary energy for actions if 'harvested'
        # also weights in the cost function, also for 'unlimited'
        send_energy = 1
        sense_energy = 0
        # 'harvested'
        B_max = 5
        h_max = 1
        # 'constrained'
        budget = decimal.Decimal(0.2)

        # sensing
        # 'random'
        new_package_prob = 0.8

        # measure
        # 'AoII'
        p_remain = 0.5
        # 'QAoI'
        query_probability = 0.5

    else:
        raise Exception('No config under this id: ' + str(config_id))

    # The sensing modes 'unlimited' and constrained are also handled using the battery model:
    # unlimited: Start with a full battery, never charge/discharge it.
    # constrained: Fill the battery with the budget every timestep.
    #              The battery must be large enough, to allow for big bursts of energy dispenses.
    # To function correctly we must set values, that are not directly associated with those modes.
    if energy_harvesting == 'unlimited':
        B_max = send_energy + sense_energy
    if energy_harvesting == 'constrained':
        B_max = 10 * (send_energy + sense_energy)  # adjust the value of 10
    if measure == 'AoII':
        new_package_prob = 1

    # check if input values are valid
    if N_process_states <= 0 or not (isinstance(N_process_states, int)):
        raise ValueError('N_process_states must be a positive int greater than 0.')
    if sensing not in ['random', 'active']:
        raise ValueError('sensing must be either \'random\' or \'active\'')
    if energy_harvesting not in ['unlimited', 'harvested', 'constrained']:
        raise ValueError('energy_harvesting must be either \'unlimited\', \'constrained\'  or \'harvested\'')
    if channel_quality <= 0 or channel_quality >= 1:
        raise ValueError('The channel quality must be between 0 and 1 (exclusive)')
    if measure not in ('AoI', 'QAoI', 'AoII'):
        raise ValueError('The measure' + str(measure) + 'is not implemented.')
    if measure == 'AoII' and sensing == 'active':
        raise ValueError('The measure AoII is not usable with sensing active. Choose sensing random instead.')
    if not isinstance(risk_sensitivity, bool):
        raise ValueError('risk_sensitivity must be a boolean.')
    if not isinstance(sense_energy, int) or not isinstance(send_energy, int) or sense_energy < 0 or send_energy < 0:
        raise ValueError('send/sense energy must be int >= 0')
    if not isinstance(B_max, int) or B_max < 0:
        raise ValueError('B_max must be int >= 0')
    if not isinstance(h_max, int) or h_max < 0:
        raise ValueError('h_max must be int >= 0')
    if new_package_prob < 0 or new_package_prob > 1:
        raise ValueError('new_package_prob must be between 0 and 1')
    if query_probability < 0 or query_probability > 1:
        raise ValueError('query_probability must be between 0 and 1')
    print('All configuration values are valid.')

    return N_process_states, sensing, energy_harvesting, channel_quality, measure, risk_sensitivity, B_max, \
        h_max, p_remain, send_energy, sense_energy, query_probability, new_package_prob, budget, threshold, risk_factor

