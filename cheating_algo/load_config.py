from typing import Generator, List, Tuple
from pathlib import Path
from json import load, dump
from datetime import datetime
from copy import deepcopy

default_values = dict(
    num_iterations=12,
    # 1: action, 2: value, 3: return both, 4: lam * action + value
    sub_iterations=[(50, 2), (50, 1)],
    end_state_values=False,
    value_optimizer="Adam",
    action_optimizer="Adam",
    value_lr=0.5,
    action_lr=0.5,
    gamma=0.8,
    eps=0.0,
    shots=100000,
    qpe_qubits=0,
    max_qpe_prob=0.8,
    output_path="./results/",
)


def _load_config(file_path: Path) -> dict:
    config = default_values
    old_file_dir = file_path.parent.resolve()
    with file_path.open("r") as f:
        config.update(load(f))
        f.close()

    datetime_str = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    output_dir = Path(config["output_path"]) / datetime_str
    config["output_dir"] = str(output_dir.resolve())

    return config


def load_config(config_path: str | Path) -> Generator[Tuple[Path, dict], None, None]:
    config_path = Path(config_path)

    if config_path.is_file():
        files = [config_path]
    else:
        files = [x for x in config_path.glob("**/*") if x.is_file() and str(x).lower().endswith(".json")]

    for f in files:
        yield f, _load_config(f)


def save_config(config_path: str | Path, config: dict):
    if isinstance(config_path, str):
        config_path = Path(config_path)
    with config_path.open("w") as f:
        dump(config, f)
        f.close()


if __name__ == "__main__":
    path = Path("./configs/default_config.json")
    save_config(path, default_values)
