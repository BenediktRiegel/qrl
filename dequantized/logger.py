from typing import List, Dict
from json import dumps as json_dump
from pathlib import Path


class Logger:
    def __init__(self, output_dir, file_name="log"):
        self.output_file = Path(output_dir) / f"{file_name}.txt"

    def log(self, iterations: int, type_itr: int, sub_iteration: int, sub_it_time: float, total_time: float, value_loss: float, action_loss: float, action_probs: Dict[str, List[float]], state_values: Dict[str, float]):
        line = json_dump(dict(iterations=iterations, type_itr=type_itr, sub_iteration=sub_iteration, sub_it_time=sub_it_time, total_time=total_time, value_loss=value_loss, action_loss=action_loss, action_probs=action_probs, state_values=state_values))
        with self.output_file.open("a") as f:
            f.write(line)
            f.close()
