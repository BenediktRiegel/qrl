from typing import List, Dict, Iterable
from json import dumps as json_dump
from json import loads
from pathlib import Path


class Logger:
    def __init__(self, output_dir, file_name="log"):
        self.output_file = Path(output_dir) / f"{file_name}.txt"

    def log(self, iterations: int, type_itr: int, sub_iteration: int, sub_it_time: float, total_time: float, value_loss: float, action_loss: float, action_probs: Iterable[Iterable[float]], state_values: Iterable[float], value_grad: Iterable[float], action_grad: Iterable[Iterable[float]]):
        line = json_dump(dict(iterations=iterations, type_itr=type_itr, sub_iteration=sub_iteration, sub_it_time=sub_it_time, total_time=total_time, value_loss=value_loss, action_loss=action_loss, action_probs=action_probs, state_values=state_values, value_grad=value_grad, action_grad=action_grad))
        with self.output_file.open("a") as f:
            f.write(line + "\n")
            f.close()


def load_log(path: Path) -> List[Dict]:
    log = []
    with path.open("r") as f:
        for line in f.readlines():
            if line:
                log.append(loads(line))
    return log
