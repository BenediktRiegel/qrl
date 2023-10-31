from typing import List, Dict, Iterable
from json import dumps as json_dump
from json import loads
from pathlib import Path


class Logger:
    """
    A class to keep a log of the training process.
    """
    def __init__(self, output_dir, file_name="log"):
        """
        Initialises logger
        :param output_dir: path to directory, in which the log file should be saved
        :param file_name: the name the log file should have
        """
        self.output_file = Path(output_dir) / f"{file_name}.txt"

    def log(self, iterations: int, type_itr: int, sub_iteration: int, sub_it_time: float, total_time: float, value_loss: float, action_loss: float, action_probs: Iterable[Iterable[float]], state_values: Iterable[float], value_grad: Iterable[float], action_grad: Iterable[Iterable[float]], action_params_change: float, value_params_change: float):
        """
        Logs parameters of the training process, by writing them into a dict and converting it into a json serializable string. Afterwards, it appends this string as a new line to the log file.
        """
        line = json_dump(dict(iterations=iterations, type_itr=type_itr, sub_iteration=sub_iteration, sub_it_time=sub_it_time, total_time=total_time, value_loss=value_loss, action_loss=action_loss, action_probs=action_probs, state_values=state_values, value_grad=value_grad, action_grad=action_grad, action_params_change=action_params_change, value_params_change=value_params_change))
        with self.output_file.open("a") as f:
            f.write(line + "\n")
            f.close()


def load_log(path: Path) -> List[Dict]:
    """
    Given the path to a log file, this method loads every line as a dict and writes it to a list.
    :param path: path to log file
    """
    log = []
    with path.open("r") as f:
        for line in f.readlines():
            if line:
                log.append(loads(line))
    return log
