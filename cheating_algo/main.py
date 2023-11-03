import torch
from pathlib import Path
from load_config import load_config
from optimizer import OptimizerEnum
from json import dump as json_save
from torch import save
from weight_init import WeightInitEnum
from train import train
from logger import Logger
from visualize import create_visualizations
from numpy import ceil
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor


def run(config_path, config, console_prints=False):
    """
    Given the path to a config and the contents of the config, this method initialises the weights of the QNNs. Further,
    it initialises the specified optimizers and a logger with the specified output directory.
    After initialising everything, it trains the parameters and save the resulting parameters, as well as the config
    in the output directory. Finally, it deletes the old config.
    :param config_path: path to config
    :param config: config as dictionary
    """
    num_iterations = config["num_iterations"]
    # 1: action, 2: value, 3: return both, 4: lam * action + value
    sub_iterations = config["sub_iterations"]
    end_state_values = config["end_state_values"]
    value_optimizer_enum = OptimizerEnum(config["value_optimizer"])
    action_optimizer_enum = OptimizerEnum(config["action_optimizer"])
    value_lr = config["value_lr"]
    action_lr = config["action_lr"]
    gamma = config["gamma"]
    eps = config["eps"]
    shots = config["shots"]
    qpe_qubits = config["qpe_qubits"]
    max_qpe_prob = config["max_qpe_prob"]
    output_dir = Path(config["output_dir"])

    output_dir.mkdir(parents=True, exist_ok=False)

    logger = Logger(output_dir)

    print("init action and value parameters")
    action_params = WeightInitEnum.standard_normal.init_params((16, 3), dtype=torch.float64)
    value_params = WeightInitEnum.standard_normal.init_params(16, dtype=torch.float64)

    value_optimizer = value_optimizer_enum.get_optimizer([value_params], value_lr)
    action_optimizer = action_optimizer_enum.get_optimizer([action_params], action_lr)

    train(value_optimizer, action_optimizer, num_iterations, sub_iterations, action_params, value_params, gamma, eps, end_state_values, shots, qpe_qubits, max_qpe_prob, logger, console_prints)

    save(action_params, output_dir / f"action_params")
    save(value_params, output_dir / f"value_qnn_param")

    with (output_dir / "config.json").open("w") as f:
        json_save(config, f)
        f.close()

    config_path.unlink()


def process_execution(worker_args):
    """
    Given an id and a list of configs, this method executes the method ``run`` with all configs.
    Additionally, it overwrites the output directory specified by the config to be the output path + the current date and time
    and the id. At the end it visualizes the results of the executed config.
    :param worker_args: tuple of an assigned id and a list of configs
    """
    process_id, configs = worker_args
    total_configs = len(configs)
    for idx, (config_path, config) in enumerate(configs):
        print(f"id {process_id}: config {idx+1}/{total_configs}")
        print(f"config_path: {config_path}\nconfig: {config}")
        # Redo output directory
        # Since we early determined the output directories at the same time, they are probably the same => We don't want that
        # Add process_id to output directory, to keep directory name unique
        sub_dir_str = datetime.now().strftime("%Y.%m.%d_%H.%M.%S") + f"_id{process_id}"
        output_dir = Path(config["output_path"]) / sub_dir_str
        config["output_dir"] = str(output_dir.resolve())
        run(config_path, config)
        create_visualizations(Path(config["output_dir"]))


def main():
    """
    Takes no input parameters. Changes have to be made directly in method.
    Given a directory containing configs, it loads the configs in the directory and one by one executes them via the method run.
    If the variable ``num_processes`` is > 1, then it starts ``num_processes`` many processes, divides the configs up fairly between them
    and executes the method ``process_execution`` for each of them
    """
    path_dir = Path("./configs/")
    num_processes = 1
    if num_processes == 1:
        from visualize import main as vis_main
        idx = 0
        for config_path, config in load_config(path_dir):
            idx += 1
            print(f"start config {idx}")
            run(config_path, config, console_prints=True)
            vis_main()
    elif num_processes > 1:
        torch.multiprocessing.set_start_method('spawn')
        # Divide work equally
        all_configs = list(load_config(path_dir))
        configs_per_process = int(ceil(len(all_configs) / num_processes))
        print(f"total configs: {len(all_configs)}")
        print(f"configs_per_process: {configs_per_process}")

        # Start processes
        ppe = ProcessPoolExecutor(max_workers=num_processes)
        worker_args = []
        for process_id in range(num_processes):
            first_config = process_id*configs_per_process
            worker_args.append((process_id, all_configs[first_config:first_config+configs_per_process]))
        results = ppe.map(process_execution, worker_args)
        for res in results:
            print(res)

    print("Finished all configs")


if __name__ == "__main__":
    main()
