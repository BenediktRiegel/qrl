from typing import Generator, List, Tuple
from pathlib import Path
from json import load, dump
from frozen_lake import FrozenField
from datetime import datetime


# Default config values
default_values = dict(
    num_iterations=12,
    # 1: action, 2: value, 3: return both, 4: lam * action + value
    sub_iterations=[(50, 2), (50, 1)],
    end_state_values=False,
    action_qnn_depth=1,
    value_qnn_depth=1,
    value_optimizer="Adam",
    action_optimizer="Adam",
    value_lr=0.5,
    action_lr=0.5,
    default_reward=0.0,
    gamma=0.8,
    eps=0.0,
    lam=0.8,
    backend="pennylane_lightning.kokkos",
    shots=100000,
    action_diff_method="best",  # best, adjoint, parameter-shift
    value_diff_method="best",   # best, adjoint, parameter-shift
    slip_probabilities=[1. / 3., 1. / 3., 0., 1. / 3.],
    map=[
        ["I", "I"],
        ["H", "G"],
    ],
    output_path="./results/",
)


def str_to_map_row(row: str):
    """
    Converts a string encoded FrozenLake map row, into one consisting out of lists.
    (Later on, this can be converted to a 2d list of FrozenFields)
    :param: str row
    :return: list new row
    """
    new_row = []
    i = 0
    while i < len(row):
        if row[i] == "(":
            j = i + 1
            while row[j] != ")":
                j += 1
            el = row[i + 1:j].split(",")
            new_row.append((el[0].strip(), el[1].strip()))

            i = j

        elif row[i].lower() == "h":
            new_row.append(row[i])
        elif row[i].lower() == "g" or row[i].lower() == "e":
            new_row.append(row[i])
        elif row[i].lower() == "i":
            new_row.append(row[i])

        i += 1

    return new_row


def load_map(config_map: List[List] | str) -> List[List[FrozenField]]:
    """
    Converts the map saved in a config file to a 2d list containing the corresponding FrozenFields
    :param config_map: 2d list | str from config file
    :return: 2d list containing FrozenFields
    """
    if isinstance(config_map, str):
        map_list = []
        rows = config_map.split("\n")
        for row in rows:
            row = row.strip()
            if row != "":
                map_list.append(str_to_map_row(row))
    frozen_lake = []
    for row in config_map:
        new_row = []
        for element in row:
            if element.lower() == "h":
                new_row.append(FrozenField.get_hole())
            elif element.lower() == "g" or element.lower() == "e":
                new_row.append(FrozenField.get_end())
            elif element.lower() == "i":
                new_row.append(FrozenField.get_ice())
            else:
                new_row.append(FrozenField(
                    reward=float(element[0]),
                    end=(element[1] or element[1] == 1)
                ))
        frozen_lake.append(new_row)
    return frozen_lake


def _load_config(file_path: Path) -> dict:
    """
    Given the path to a config, i.e. a .json file, this method will load the contents of the config into a dictionary.
    Missing values will be replaced by the default values. Additionally, it adds another folder to the directory
    specified by the entry 'ouput_dir' of the config. The name of this folder is the current date and time.
    :param file_path: path to config file
    :return: dictionary of config
    """
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
    """
    Takes a path to either a directory containing configs or a single config.
    Returns a generator that loads each config and returns it with its path
    :param config_path: path to single config or a directory containing configs
    :return: generator each yield gives the file_path to the next config and the next config itself.
    """
    config_path = Path(config_path)

    if config_path.is_file():
        files = [config_path]
    else:
        files = [x for x in config_path.glob("**/*") if x.is_file() and str(x).lower().endswith(".json")]

    for f in files:
        yield f, _load_config(f)


def save_config(config_path: str | Path, config: dict):
    """
    Dumps config into a .json file
    :param config_path: path to a .json file in which to save the config.
    """
    if isinstance(config_path, str):
        config_path = Path(config_path)
    with config_path.open("w") as f:
        dump(config, f)
        f.close()


if __name__ == "__main__":
    path = Path("./configs/default_config.json")
    save_config(path, default_values)
