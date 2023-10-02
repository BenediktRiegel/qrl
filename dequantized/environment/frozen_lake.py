from typing import List, Tuple
import torch
import numpy as np
from utils import int_to_bitlist, bitlist_to_int


class FrozenField:
    def __init__(self, reward: float = None, end: bool = False):
        self.reward = reward
        self.end = end

    @staticmethod
    def get_hole():
        return FrozenField(reward=0, end=True)

    @staticmethod
    def get_end():
        return FrozenField(reward=1, end=True)

    @staticmethod
    def get_ice():
        return FrozenField(reward=None, end=False)


class FrozenLake:
    def __init__(
            self, map: List[List[FrozenField]], slip_probabilities: List[float], default_reward: float = 0.,
    ):
        self.map = map
        self.slip_probabilities = slip_probabilities
        self.default_reward = default_reward

        self.r_m = None
        for row in map:
            for field in row:
                if field.reward is not None:
                    if self.r_m is None:
                        self.r_m = np.abs(field.reward)
                    else:
                        self.r_m = max(np.abs(field.reward), self.r_m)
        if self.r_m is None:
            self.r_m = np.abs(self.default_reward)
        else:
            self.r_m = max(np.abs(self.default_reward), self.r_m)
        if self.r_m != 1 and self.r_m != 0:
            for y, row in enumerate(map):
                for x, field in enumerate(row):
                    if field.reward is not None:
                        map[y][x].reward = field.reward / self.r_m

        self.default_reward /= self.r_m

        self.x_bits = int(np.ceil(np.log2(len(self.map[0]))))
        self.y_bits = int(np.ceil(np.log2(len(self.map))))
        self.a_bits = 2

    def sample_transition(self, s, a) -> Tuple[List[List[int]], List[int], float]:
        threshold = torch.rand(1)
        current_p = 0
        a = int_to_bitlist(a, self.a_bits)
        slip_a = 0
        for idx, p in enumerate(self.slip_probabilities):
            current_p += p
            if threshold <= current_p:
                slip_a = (idx + a) % 4

        x = bitlist_to_int(s[0])
        y = bitlist_to_int(s[1])
        (x_, y_, r) = self._transistion(x, y, slip_a)
        return [int_to_bitlist(x_, self.x_bits), int_to_bitlist(y_, self.y_bits)], r

    def _transistion(self, x, y, a) -> Tuple[int, int, float]:
        if a == 0:
            if x < len(self.map) - 1:
                x += 1
        elif a == 1:
            if y != 0:
                y -= 1
        elif a == 2:
            if x != 0:
                x -= 1
        elif a == 3:
            if y < len(self.map) - 1:
                y += 1

        r = self.map[y][x].reward
        if r is None:
            r = self.default_reward
        return x, y, r

    def get_random_states(self, num_states: int):
        y = torch.tensor([int_to_bitlist(el, self.y_bits) for el in torch.randint(0, len(self.map), (num_states,))])
        x = torch.tensor([int_to_bitlist(el, self.x_bits) for el in torch.randint(0, len(self.map[0]), (num_states,))])
        return torch.vstack((x, y)).T
