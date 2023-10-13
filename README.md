# Quantum Reinforcement Learning

This project is part of the master thesis "Quantum Reinforcement Learning using Entangled States" by Benedikt Riegel. The examiner is Prof. Dr. h. c. Frank Leymann and the Supervisors are Dipl.-Ing. Alexander Mandl and Marvin Bechthold, M.Sc. at the institute of architecture of application systems at the university Stuttgart.


## Getting Started
To get started, please follow these steps:
1. Clone the repository to your local machine
2. Make sure you have Python 3.10 and poetry installed
3. Go into the directory ``qrl/`` and run the command ``poetry install``
4. Go into the directory ``qrl/pennylane_implementation`` or ``qrl/cheating_algo``
5. Move the configs of your choice into the directory ``configs/``
6. Run ``poetry run python3 main.py``

This will run all configs within the ``configs/`` directory.

Note: 
After successfully executing the algorithm with a given config, the config will be moved to the folder containing the results. If you want to reuse the config, please make sure to correct the hyperparameter ``output_dir``, since this parameter will be changed. We suggest having a ``.txt``-file as a copy of the config to quickly reuse/-purpose the config.

The directory ``configs/`` should already contain example configurations. To use them, please change the file ending from ``.txt`` to ``.json``. If you use these configs, please make sure that there is a directory called ``results/`` within the algorithm directory, i.e. on the same height of the directory hierachy as ``configs/``.


## Configs
A config is a ``.json``-file, containing a dictionary with all the Hyperparameters used to run the algorithm. The directory ``configs/`` should contain some example configs by default, except that they have a ``.txt`` ending, instead of ``.json``.


## Visualizations
Currently ``qrl/pennylane_implementation`` creates the visualizations live during runtime and ``qrl/cheating_algo`` creates them afterwards.
