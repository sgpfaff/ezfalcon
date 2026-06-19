import numpy as np
from dataclasses import dataclass
from .potential import HaloModel

def create_r_array(R_max, N):
    dr = R_max / N
    r_array  = (np.arange(N) + 0.5) * dr   # half-integer grid: r[0] = dr/2
    return r_array

@dataclass
class Solution:
    solution_dict: dict
    HaloModel: object
    r_array: np.ndarray
    