from pathlib import Path
from numpy import ceil, log2
from torch import save
from json import dump as json_save
from torch.optim.lr_scheduler import StepLR
from optimizer import OptimizerEnum
from quantum_backends import QuantumBackends
from train import train_with_two_opt
from qnns import *
from qnns import WeightInitEnum
from frozen_lake import FrozenLake
from loss import loss_function as loss
from wire_utils import get_wires
from visualize import get_action_probs, get_frozen_lake_frame, plot_animated_frozen_lake, plot_loss
from load_config import load_config, load_map


def main(config_path: Path, config: dict):
    """
    Given a config and its path, this method runs the Quantum Policy Iteration with the specified config parameters.
    At the end it saves the results and visualizations in the from the config specified path. It also copies the config
    and deletes it in its old path.
    :param config_path: Path
    :param config: dict
    """
    num_iterations = config["num_iterations"]
    # 1: action, 2: value, 3: return both, 4: lam * action + value
    sub_iterations = config["sub_iterations"]
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

    fig_path = output_dir / "fig.html"
    loss_path = output_dir / "loss.html"
    print("prepare environment")
    map = [el for el in reversed(map)]
    environment = FrozenLake(map, slip_probabilities, default_reward=default_reward, r_qubit_is_clean=True)

    log_rows = int(ceil(log2(len(map))))
    log_cols = int(ceil(log2(len(map[0]))))

    # Loss function
    # wires, total_num_wires = get_wires([log_cols, log_rows, 2, log_cols, log_rows, len(r_qubit_interpretation), len(r_qubit_interpretation)+3, len(r_qubit_interpretation)+3])
    # x_qubits, y_qubits, action_qubits, next_x_qubits, next_y_qubits, r_qubits, value_qubits, next_value_qubits = wires
    # Loss function2
    # wires, total_num_wires = get_wires([log_cols, log_rows, 2, log_cols, log_rows, 10])
    wires, total_num_wires = get_wires([log_cols, log_rows, 2, log_cols, log_rows, 7])
    x_qubits, y_qubits, action_qubits, next_x_qubits, next_y_qubits, ancilla_qubits = wires
    backend = backend_enum.get_pennylane_backend(total_num_wires, shots)
    print(f"{total_num_wires} qubits")

    print("prepare qnn")
    # action_qnn = QNN(len(x_qubits), len(action_qubits), action_qnn_depth)
    action_qnn = RYQNN_Excessive(len(x_qubits) + len(y_qubits), len(action_qubits), action_qnn_depth,
                                 WeightInitEnum.standard_normal)
    # action_qnn.in_q_parameters = load("./action_qnn/param0")
    value_qnn = CCRYQNN_Excessive(len(x_qubits) + len(y_qubits), value_qnn_depth, WeightInitEnum.standard_normal)
    # value_qnn.in_q_parameters = load("./value_qnn/param0")
    # action_qnn = RYQNN_D(len(x_qubits) + len(y_qubits), len(action_qubits), action_qnn_depth, WeightInitEnum.standard_normal)
    # value_qnn = CCRYQNN_D(len(x_qubits) + len(y_qubits), value_qnn_depth, WeightInitEnum.standard_normal)
    print(f"value_qnn parameters: ")
    for p in value_qnn.parameters():
        print(p)
    # optimizer = optimizer.get_optimizer(action_qnn.parameters() + value_qnn.parameters(), lr)
    value_optimizer = value_optimizer_enum.get_optimizer(value_qnn.parameters(), value_lr)
    value_scheduler = StepLR(value_optimizer, step_size=1, gamma=2.)
    action_optimizer = action_optimizer_enum.get_optimizer(action_qnn.parameters(), action_lr)
    action_scheduler = StepLR(value_optimizer, step_size=1, gamma=2.)
    loss_fn = loss

    # fig = plot_frozen_lake(environment, action_qnn, len(x_qubits), len(y_qubits))
    # fig.show()

    loss_function_params = dict(
        environment=environment,
        x_qubits=x_qubits, y_qubits=y_qubits, action_qubits=action_qubits,
        next_x_qubits=next_x_qubits, next_y_qubits=next_y_qubits,
        ancilla_qubits=ancilla_qubits,
        unclean_qubits=[],
        backend=backend,
        gamma=gamma, lam=lam,
        eps=eps,
        end_state_values=end_state_values,
        action_diff_method=action_diff_method,
        value_diff_method=value_diff_method,
    )

    frames, losses = train_with_two_opt(
        loss_fn, value_optimizer, action_optimizer, value_scheduler, action_scheduler,
        num_iterations, sub_iterations, action_qnn, value_qnn,
        loss_function_params, fig_path, loss_path
    )
    frames = frames + [get_frozen_lake_frame(environment, action_qnn, value_qnn, len(x_qubits), len(y_qubits), gamma,
                                    end_state_values)]

    for param in action_qnn.parameters():
        param.requires_grad = False
    temp = []
    for y in range(len(map)):
        temp.append([])
        for x in range(len(map[0])):
            probs = get_action_probs(x, y, action_qnn, len(x_qubits), len(y_qubits))
            temp[-1].append({"right": probs[0], "down": probs[1], "left": probs[2], "up": probs[3]})

    print(f"True movements:")
    print([el for el in reversed(temp)])

    for param in action_qnn.parameters():
        param.requires_grad = True

    for i, param in enumerate(action_qnn.parameters()):
        save(param, output_dir / f"action_qnn_param{i}")
    for i, param in enumerate(value_qnn.parameters()):
        save(param, output_dir / f"value_qnn_param{i}")

    fig = plot_animated_frozen_lake(environment, frames, gamma)
    with fig_path.open("w", encoding="utf-8") as f:
        f.write(fig.to_html())
        f.close()

    fig = plot_loss(losses)
    with loss_path.open("w", encoding="utf-8") as f:
        f.write(fig.to_html())
        f.close()

    with (output_dir / "config.json").open("w") as f:
        json_save(config, f)
        f.close()

    config_path.unlink()


if __name__ == "__main__":
    # Execute all configs in path "./configs" one after another
    path_dir = Path("./configs/")
    for config_path, config in load_config(path_dir):
        main(config_path, config)
