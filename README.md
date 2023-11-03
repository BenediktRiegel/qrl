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

### pennylane_implementation
A config my consist of the following parameters:
- num_iteration: Integer that determines how often the sub_iterations will be repeated
- sub_iterations: List of List, e.g. ``[[25, 2], [12, 1]]``, these list contains two sub-iterations the first is ``[25, 2]`` the 2 represents that the value QNN will be updated, meanwhile the 25 tells us how often. If we have instead of 2, we update the action QNN.
- end_state_values: If true, then terminal states also learn a state value, otherwise they do not. In our thesis we allowed terminal states to learn a state value.
- action_qnn_depth: How many Layers the action QNN has
- value_qnn_depth: How many Layers the value QNN has
- value_optimizer: Determines which optimizer should be used to update the values QNN, we suggest ``Adam``, but more can be found in optimizer.py.
- action_optimizer: Determines which optimizer should be used to update the action QNN, we suggest ``Adam``, but more can be found in optimizer.py.
- value_lr: Learning rate for the value_optimizer
- action_lr: Learning rate for the action_optimizer
- gamma: Discount factor
- shots: Number of shots
- map: Determines the map of the FrozenLake, e.g. ``[["I", "I"], ["H", "G"]]``. I's are normal fields, H's are holes and G's are goal states.
- output_path: Path were the output should be saved to

Note: Only parameters listed here, where tested!

### cheating_algo
A config my consist of the following parameters:
- num_iteration: Integer that determines how often the sub_iterations will be repeated
- sub_iterations: List of List, e.g. ``[[0.0001, 25, 2], [0.0001, 25, 1]]``, these list contains two sub-iterations the first is ``[0.0001, 25, 2]`` the 2 represents that the value QNN will be updated, meanwhile the 25 tells us how often. If the updates are less than 0.0001 big, then we stop this sub_itr and continue with the next, e.g. if the value QNN has a too small update, we jump to updating the action QNN. If we have instead of 2, we update the action QNN.
- value_optimizer: Determines which optimizer should be used to update the values QNN, we suggest ``Adam``, but more can be found in optimizer.py.
- action_optimizer: Determines which optimizer should be used to update the action QNN, we suggest ``Adam``, but more can be found in optimizer.py.
- value_lr: Learning rate for the value_optimizer
- action_lr: Learning rate for the action_optimizer
- gamma: Discount factor
- shots: Number of shots
- qpe_qubits: Number of qubits used for the QPE estimation
- max_qpe_prob: Probability threshold for the QPE estimation process
- output_path: Path were the output should be saved to

If shots is null, then the exact loss will be used to train. If shots is not None and qpe_qubits is 0, then we simple sample. Lastly, if the shots are not None and the qpe_qubits are greater than zero, then we use QPE estimation.

A method to automatically create configs can be found in ``create_configs.py``.

Note: Only parameters listed here, where tested!

## Visualizations
Currently ``qrl/pennylane_implementation`` creates the visualizations live during runtime and ``qrl/cheating_algo`` creates them afterwards.
