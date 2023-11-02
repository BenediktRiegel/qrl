from json import dump as json_dump


def main():
    """
    Creates configs in the directory ./configs
    The method determines lists for the different parameters.
    This method will create a config for each parameter combination given by these lists.
    """
    output_path = "./results/"
    output_dir = "./results/hyperparameter_search"
    end_state_values = True
    # 'num_iterations', 'sub_iterations', 'value_optimizer', 'action_optimizer', 'value_lr', 'action_lr', 'gamma', 'eps', 'shots', 'qpe_qubits', 'max_qpe_prob'
    # num_iterations_list = [12, 24]
    # sub_iterations_list = [[[0.0001, 25, 2], [0.0001, 25, 1]], [[0.0001, 50, 2], [0.0001, 50, 1]], [[0.0001, 100, 2], [0.0001, 100, 1]]]
    # value_optimizer_list = ["Adam"]
    # action_optimizer_list = ["Adam"]
    # value_lr_list = [0.5, 0.25]
    # action_lr_list = [0.5, 0.25]
    # gamma_list = [0.8, 0.9]
    # eps_list = [0.0]
    # shots_list = [10, 50, 100, 1000, 10000, 100000, 1000000, None]
    # qpe_qubits_list = [0, 8, 16, 32, 64]
    # max_qpe_prob_list = [0.8]
    num_iterations_list = [12]
    sub_iterations_list = [[[0.0001, 25, 2], [0.0001, 25, 1]]]
    value_optimizer_list = ["Adam"]
    action_optimizer_list = ["Adam"]
    value_lr_list = [0.5]
    action_lr_list = [0.5]
    gamma_list = [0.8]
    eps_list = [0.0]
    # shots_list = [10, 50, 100, 1000, 10000, 100000, 1000000, None]
    # qpe_qubits_list = [0, 8, 16, 32, 64]
    shots_list = [10000000]
    qpe_qubits_list = [0]
    max_qpe_prob_list = [0.8]

    num_repetitions = 1

    num = 1
    for _ in range(num_repetitions):
        for num_iterations in num_iterations_list:
            for sub_iterations in sub_iterations_list:
                for value_optimizer in value_optimizer_list:
                    for action_optimizer in action_optimizer_list:
                        for value_lr, action_lr in zip(value_lr_list, action_lr_list):
                            # for action_lr in action_lr_list:
                            for gamma in gamma_list:
                                for eps in eps_list:
                                    for shots in shots_list:
                                        for qpe_qubits in qpe_qubits_list:
                                            for max_qpe_prob in max_qpe_prob_list:
                                                # if qpe_qubits > 0 and (not shots or shots > 100):
                                                #     continue
                                                config = dict(
                                                    num_iterations=num_iterations,
                                                    sub_iterations=sub_iterations,
                                                    end_state_values=end_state_values,
                                                    value_optimizer=value_optimizer,
                                                    action_optimizer=action_optimizer,
                                                    value_lr=value_lr,
                                                    action_lr=action_lr,
                                                    gamma=gamma,
                                                    eps=eps,
                                                    shots=shots,
                                                    qpe_qubits=qpe_qubits,
                                                    max_qpe_prob=max_qpe_prob,
                                                    output_dir=output_dir,
                                                    output_path=output_path,
                                                )
                                                with open(f"./configs/config{num}.json", "w") as f:
                                                    json_dump(config, f)
                                                    f.close()
                                                num += 1
    print(f"Created {num-1} configs")


if __name__ == "__main__":
    main()
