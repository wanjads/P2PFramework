Copyright (c) 2023 wanjads, F3Pyttel

This code implements the algorithm proposed in the paper "A Unified Approach to Learn Transmission Strategies Using Age-Based Metrics in Point-to-Point Wireless Communication".
It learns and tests transmission strategies for age-based metrics in point-to-point scenarios.

To create a configuration copy an elif statement in the load function of configs.py.
Change the config id to an unused value and adjust the config variables.
Not all variables are needed for every main-mode (defined by sensing, energy-harvesting and measure). Leave dummy values
in those variables. They will not be used but must be assigned.

To start a simulation enter the config id and further simulation parameters in the file parameters.py
and start the simulation by executing main.py. Basic results are printed.

To get more insight you can plot via the plotting module. Import the file plotting.py and use the available functions.
