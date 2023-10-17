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


def main(config_path, config):
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
    # import torch
    # # right: 000, down: 010, left 100, up: 110
    # action_params = torch.tensor([
    #     [0, 0, 0],
    #     [0, 0, 0],
    #     [0, 0, 0],
    #     [0, 0, 0],
    #
    #     [1, 1, 0],
    #     [0, 1, 0],
    #     [1, 0, 0],
    #     [0, 0, 0],
    #
    #     [1, 0, 0],
    #     [0, 0, 0],
    #     [0, 0, 0],
    #     [0, 0, 0],
    #
    #     [1, 0, 0],
    #     [1, 1, 0],
    #     [1, 0, 0],
    #     [1, 1, 0],
    # ])*torch.pi
    # value_params = WeightInitEnum.zero.init_params(16, dtype=torch.float64) + torch.pi
    value_params = WeightInitEnum.standard_normal.init_params(16, dtype=torch.float64)
    # from torch import load
    # action_params = load("./results/2023.10.07_11.12.11/action_params")
    # value_params = load("./results/2023.10.07_11.12.11/value_qnn_param")
    # action_params.requires_grad = True
    # value_params.requires_grad = True

    value_optimizer = value_optimizer_enum.get_optimizer([value_params], value_lr)
    action_optimizer = action_optimizer_enum.get_optimizer([action_params], action_lr)

    train(value_optimizer, action_optimizer, num_iterations, sub_iterations, action_params, value_params, gamma, eps, end_state_values, shots, qpe_qubits, max_qpe_prob, logger)

    save(action_params, output_dir / f"action_params")
    save(value_params, output_dir / f"value_qnn_param")

    with (output_dir / "config.json").open("w") as f:
        json_save(config, f)
        f.close()

    config_path.unlink()


def process_execution(worker_args):
    process_id, configs = worker_args
    for config_path, config in configs:
        # Redo output directory
        # Since we early determined the output directories at the same time, they are probably the same => We don't want that
        # Add process_id to output directory, to keep directory name unique
        sub_dir_str = datetime.now().strftime("%Y.%m.%d_%H.%M.%S") + f"_id{process_id}"
        output_dir = Path(config["output_path"]).parent / sub_dir_str
        config["output_dir"] = str(output_dir.resolve())
        main(config_path, config)
        create_visualizations(Path(config["output_dir"]))


if __name__ == "__main__":
    path_dir = Path("./configs/")
    num_processes = 8
    if num_processes == 1:
        from visualize import main as vis_main
        for config_path, config in load_config(path_dir):
            main(config_path, config)
            vis_main()
    elif num_processes > 1:
        torch.multiprocessing.set_start_method('spawn')
        # Divide work equally
        all_configs = list(load_config(path_dir))
        configs_per_process = int(ceil(len(all_configs) / num_processes))

        # Start processes
        ppe = ProcessPoolExecutor(max_workers=num_processes)
        worker_args = []
        for process_id in range(num_processes):
            first_config = process_id*configs_per_process
            worker_args.append((process_id, all_configs[first_config:first_config+configs_per_process]))
        results = ppe.map(process_execution, worker_args)
        for res in results:
            print(res)
