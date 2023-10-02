from pathlib import Path
from numpy import ceil, log2
from torch import save
from json import dump as json_save
from torch.optim.lr_scheduler import StepLR
from optimizer import OptimizerEnum
from quantum_backends import QuantumBackends
from train import train
from qnns.table_like_qnns import RYQNN_Excessive
from qnns.weight_init import WeightInitEnum
from environment.frozen_lake import FrozenLake
from loss_function import loss_function
from wire_utils import get_wires
from visualize.rot_swap_value import get_action_probs, get_frozen_lake_frame, plot_animated_frozen_lake, plot_loss
from load_config import load_config, load_map
from logger import Logger


def main(config_path: Path, config: dict):
    """
    Prepare input from config
    """
    num_iterations = config["num_iterations"]
    # 1: action, 2: value, 3: return both, 4: lam * action + value
    sub_iterations = config["sub_iterations"]
    precise = config["precise"]
    end_state_values = config["end_state_values"]
    action_qnn_depth = config["action_qnn_depth"]
    value_qnn_depth = config["value_qnn_depth"]
    value_optimizer_enum = OptimizerEnum(config["value_optimizer"])
    action_optimizer_enum = OptimizerEnum(config["action_optimizer"])
    value_lr = config["value_lr"]
    action_lr = config["action_lr"]
    default_reward = config["default_reward"]
    gamma = config["gamma"]
    eps = config["eps"]
    lam = config["lam"]
    backend_enum = QuantumBackends(config["backend"])
    shots = config["shots"]
    action_diff_method = config["action_diff_method"]  # best, adjoint, parameter-shift
    value_diff_method = config["value_diff_method"]  # best, adjoint, parameter-shift
    slip_probabilities = config["slip_probabilities"]
    map = load_map(config["map"])
    output_dir = Path(config["output_dir"])

    output_dir.mkdir(parents=True, exist_ok=False)

    logger = Logger(output_dir)

    fig_path = output_dir / "fig.html"
    loss_path = output_dir / "loss.html"
    print("prepare environment")
    map = [el for el in reversed(map)]
    environment = FrozenLake(map, slip_probabilities, default_reward=default_reward)

    num_x_qubits = int(ceil(log2(len(map))))
    num_y_qubits = int(ceil(log2(len(map[0]))))
    num_action_qubits = 2

    # Loss function
    print("prepare qnn")
    action_qnn = RYQNN_Excessive(num_x_qubits + num_y_qubits, num_action_qubits, action_qnn_depth,
                                 WeightInitEnum.standard_normal)
    value_qnn = RYQNN_Excessive(num_x_qubits + num_y_qubits, 1, value_qnn_depth, WeightInitEnum.standard_normal)

    action_additional_qubits, action_ancilla_qubits = action_qnn.num_additional_and_ancilla_qubits()
    value_additional_qubits, value_ancilla_qubits = value_qnn.num_additional_and_ancilla_qubits()
    print(f"value_qnn parameters: ")
    for p in value_qnn.parameters():
        print(p)

    value_wires, value_num_wires = get_wires([1, 3, 3, value_additional_qubits, value_additional_qubits, max(0, value_ancilla_qubits - 1)])
    action_wires, action_num_wires = get_wires([2, action_additional_qubits, action_ancilla_qubits])
    value_backend = backend_enum.get_pennylane_backend("", "", value_num_wires, 1)
    action_backend = backend_enum.get_pennylane_backend("", "", action_num_wires, 1)
    print(f"{value_num_wires} value qubits")
    print(f"{action_num_wires} action qubits")

    # optimizer = optimizer.get_optimizer(action_qnn.parameters() + value_qnn.parameters(), lr)
    value_optimizer = value_optimizer_enum.get_optimizer(value_qnn.parameters(), value_lr)
    value_scheduler = StepLR(value_optimizer, step_size=1, gamma=2.)
    action_optimizer = action_optimizer_enum.get_optimizer(action_qnn.parameters(), action_lr)
    action_scheduler = StepLR(value_optimizer, step_size=1, gamma=2.)
    loss_fn = loss_function

    # fig = plot_frozen_lake(environment, action_qnn, len(x_qubits), len(y_qubits))
    # fig.show()

    loss_function_params = dict(
        environment=environment,
        unclean_qubits=[],
        value_qubits=value_wires,
        action_qubits=action_wires,
        value_backend=value_backend,
        action_backend=action_backend,
        shots=shots,
        gamma=gamma, lam=lam,
        eps=eps,
        precise=precise,
        end_state_values=end_state_values,
        action_diff_method=action_diff_method,
        value_diff_method=value_diff_method,
    )

    frames, losses = train(
        loss_fn, value_optimizer, action_optimizer, value_scheduler, action_scheduler,
        num_iterations, sub_iterations, action_qnn, value_qnn,
        loss_function_params, fig_path, loss_path, logger
    )
    # frames = frames + [get_frozen_lake_frame(environment, action_qnn, value_qnn, len(x_qubits), len(y_qubits), gamma,
    #                                          end_state_values)]

    for param in action_qnn.parameters():
        param.requires_grad = False
    temp = []
    for y in range(len(map)):
        temp.append([])
        for x in range(len(map[0])):
            probs = get_action_probs(x, y, action_qnn, num_x_qubits, num_y_qubits)
            temp[-1].append({"right": probs[0], "down": probs[1], "left": probs[2], "up": probs[3]})

    print(f"True movements:")
    print([el for el in reversed(temp)])

    for param in action_qnn.parameters():
        param.requires_grad = True

    for i, param in enumerate(action_qnn.parameters()):
        save(param, output_dir / f"action_qnn_param{i}")
    for i, param in enumerate(value_qnn.parameters()):
        save(param, output_dir / f"value_qnn_param{i}")

    # fig = plot_animated_frozen_lake(environment, frames, gamma)
    # with fig_path.open("w", encoding="utf-8") as f:
    #     f.write(fig.to_html())
    #     f.close()

    fig = plot_loss(losses)
    with loss_path.open("w", encoding="utf-8") as f:
        f.write(fig.to_html())
        f.close()

    with (output_dir / "config.json").open("w") as f:
        json_save(config, f)
        f.close()

    config_path.unlink()


if __name__ == "__main__":
    path_dir = Path("./configs/")
    for config_path, config in load_config(path_dir):
        main(config_path, config)
