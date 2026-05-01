from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from nr_phy_simu.common.interfaces import DmrsSequenceGenerator
from nr_phy_simu.common.sequences.dmrs_tables import resolve_dmrs_symbol_indices
from nr_phy_simu.config import SimulationConfig


SHORT_LOW_PAPR_TYPE1_PHASES: dict[int, dict[int, tuple[int, ...]]] = {
    6: {
        0: (-3, -1, 3, 3, -1, -3), 1: (-3, 3, -1, -1, 3, -3), 2: (-3, -3, -3, 3, 1, -3),
        3: (1, 1, 1, 3, -1, -3), 4: (1, 1, 1, -3, -1, 3), 5: (-3, 1, -1, -3, -3, -3),
        6: (-3, 1, 3, -3, -3, -3), 7: (-3, -1, 1, -3, 1, -1), 8: (-3, -1, -3, 1, -3, -3),
        9: (-3, -3, 1, -3, 3, -3), 10: (-3, 1, 3, 1, -3, -3), 11: (-3, -1, -3, 1, 1, -3),
        12: (1, 1, 3, -1, -3, 3), 13: (1, 1, 3, 3, -1, 3), 14: (1, 1, 1, -3, 3, -1),
        15: (1, 1, 1, -1, 3, -3), 16: (-3, -1, -1, -1, 3, -1), 17: (-3, -3, -1, 1, -1, -3),
        18: (-3, -3, -3, 1, -3, -1), 19: (-3, 1, 1, -3, -1, -3), 20: (-3, 3, -3, 1, 1, -3),
        21: (-3, 1, -3, -3, -3, -1), 22: (1, 1, -3, 3, 1, 3), 23: (1, 1, -3, -3, 1, -3),
        24: (1, 1, 3, -1, 3, 3), 25: (1, 1, -3, 1, 3, 3), 26: (1, 1, -1, -1, 3, -1),
        27: (1, 1, -1, 3, -1, -1), 28: (1, 1, -1, 3, -3, -1), 29: (1, 1, -3, 1, -1, -1),
    },
    12: {
        0: (-3, 1, -3, -3, -3, 3, -3, -1, 1, 1, 1, -3),
        1: (-3, 1, -3, 3, 1, -3, 1, 3, -1, -1, 1, 3),
        2: (-3, 3, 3, 1, -3, 3, -1, 1, 3, -3, 3, -3),
        3: (-3, -3, -1, 3, 3, 3, -3, 3, -3, 1, -1, -3),
        4: (-3, -1, -1, 1, 3, 1, 1, -1, 1, -1, -3, 1),
        5: (-3, -3, 3, 1, -3, -3, -3, -1, 3, -1, 1, 3),
        6: (1, -1, 3, -1, -1, -1, -3, -1, 1, 1, 1, -3),
        7: (-1, -3, 3, -1, -3, -3, -3, -1, 1, -1, 1, -3),
        8: (-3, -1, 3, 1, -3, -1, -3, 3, 1, 3, 3, 1),
        9: (-3, -1, -1, -3, -3, -1, -3, 3, 1, 3, -1, -3),
        10: (-3, 3, -3, 3, 3, -3, -1, -1, 3, 3, 1, -3),
        11: (-3, -1, -3, -1, -1, -3, 3, 3, -1, -1, 1, -3),
        12: (-3, -1, 3, -3, -3, -1, -3, 1, -1, -3, 3, 3),
        13: (-3, 1, -1, -1, 3, 3, -3, -1, -1, -3, -1, -3),
        14: (1, 3, -3, 1, 3, 3, 3, 1, -1, 1, -1, 3),
        15: (-3, 1, 3, -1, -1, -3, -3, -1, -1, 3, 1, -3),
        16: (-1, -1, -1, -1, 1, -3, -1, 3, 3, -1, -3, 1),
        17: (-1, 1, 1, -1, 1, 3, 3, -1, -1, -3, 1, -3),
        18: (-3, 1, 3, 3, -1, -1, -3, 3, 3, -3, 3, -3),
        19: (-3, -3, 3, -3, -1, 3, 3, 3, -1, -3, 1, -3),
        20: (3, 1, 3, 1, 3, -3, -1, 1, 3, 1, -1, -3),
        21: (-3, 3, 1, 3, -3, 1, 1, 1, 1, 3, -3, 3),
        22: (-3, 3, 3, 3, -1, -3, -3, -1, -3, 1, 3, -3),
        23: (3, -1, -3, 3, -3, -1, 3, 3, 3, -3, -1, -3),
        24: (-3, -1, 1, -3, 1, 3, 3, 3, -1, -3, 3, 3),
        25: (-3, 3, 1, -1, 3, 3, -3, 1, -1, 1, -1, 1),
        26: (-1, 1, 3, -3, 1, -1, 1, -1, -1, -3, 1, -1),
        27: (-3, -3, 3, 3, 3, -3, -1, 1, -3, 3, 1, -3),
        28: (1, -1, 3, 1, 1, -1, -1, -1, 1, 3, -3, 1),
        29: (-3, 3, -3, 3, -3, -3, 3, -1, -1, 1, 3, -3),
    },
    18: {
        0: (-1, 3, -1, -3, 3, 1, -3, -1, 3, -3, -1, -1, 1, 1, 1, -1, -1, -1),
        1: (1, 3, -3, 3, -1, 1, 3, -3, -1, -3, -3, -1, -3, 3, 1, -1, 3, -3),
        2: (-3, 3, 1, -1, -1, 3, -3, -1, 1, 1, 1, 1, 1, -1, 3, -1, -3, -1),
        3: (3, -3, 3, 3, 3, 1, -3, 1, 3, 3, 1, -3, -3, 3, -1, -3, -1, 1),
        4: (1, 1, -1, -1, -3, -1, 1, -3, -3, -3, 1, -3, -1, -1, 1, -1, 3, 1),
        5: (3, -3, 1, 1, 3, -1, 1, -1, -1, -3, 1, 1, -1, 3, 3, -3, 3, -1),
        6: (-3, 3, -1, 1, 3, 1, -3, -1, 1, 1, -3, 1, 3, 3, -1, -3, -3, -3),
        7: (1, 1, -3, 3, 3, 1, 3, -3, 3, -1, 1, 1, -1, 1, -3, -3, -1, 3),
        8: (-3, 1, -3, -3, 1, -3, -3, 3, 1, -3, -1, -3, -3, -3, -1, 1, 1, 3),
        9: (3, -1, 3, 1, -3, -3, -1, 1, -3, -3, 3, 3, 3, 1, 3, -3, 3, -3),
        10: (-3, -3, -3, 1, -3, 3, 1, 1, 3, -3, -3, 1, 3, -1, 3, -3, -3, 3),
        11: (-3, -3, 3, 3, 3, -1, -1, -3, -1, -1, -1, 3, 1, -3, -3, -1, 3, -1),
        12: (-3, -1, -3, -3, 1, 1, -1, -3, -1, -3, -1, -1, 3, 3, -1, 3, 1, 3),
        13: (1, 1, -3, -3, -3, -3, 1, 3, -3, 3, 3, 1, -3, -1, 3, -1, -3, 1),
        14: (-3, 3, -1, -3, -1, -3, 1, 1, -3, -3, -1, -1, 3, -3, 1, 3, 1, 1),
        15: (3, 1, -3, 1, -3, 3, 3, -1, -3, -3, -1, -3, -3, 3, -3, -1, 1, 3),
        16: (-3, -1, -3, -1, -3, 1, 3, -3, -1, 3, 3, 3, 1, -1, -3, 3, -1, -3),
        17: (-3, -1, 3, 3, -1, 3, -1, -3, -1, 1, -1, -3, -1, -1, -1, 3, 3, 1),
        18: (-3, 1, -3, -1, -1, 3, 1, -3, -3, -3, -1, -3, -3, 1, 1, 1, -1, -1),
        19: (3, 3, 3, -3, -1, -3, -1, 3, -1, 1, -1, -3, 1, -3, -3, -1, 3, 3),
        20: (-3, 1, 1, -3, 1, 1, 3, -3, -1, -3, -1, 3, -3, 3, -1, -1, -1, -3),
        21: (1, -3, -1, -3, 3, 3, -1, -3, 1, -3, -3, -1, -3, -1, 1, 3, 3, 3),
        22: (-3, -3, 1, -1, -1, 1, 1, -3, -1, 3, 3, 3, 3, -1, 3, 1, 3, 1),
        23: (3, -1, -3, 1, -3, -3, -3, 3, 3, -1, 1, -3, -1, 3, 1, 1, 3, 3),
        24: (3, -1, -1, 1, -3, -1, -3, -1, -3, -3, -1, -3, 1, 1, 1, -3, -3, 3),
        25: (-3, -3, 1, -3, 3, 3, 3, -1, 3, 1, 1, -3, -3, -3, 3, -3, -1, -1),
        26: (-3, -1, -1, -3, 1, -3, 3, -1, -1, -3, 3, 3, -3, -1, 3, -1, -1, -1),
        27: (-3, -3, 3, 3, -3, 1, 3, -1, -3, 1, -1, -3, 3, -3, -1, -1, -1, 3),
        28: (-1, -3, 1, -3, -3, -3, 1, 1, 3, 3, -3, 3, 3, -3, -1, 3, -3, 1),
        29: (-3, 3, 1, -1, -1, -1, -1, 1, -1, 3, 3, -3, -1, 1, 3, -1, 3, -1),
    },
    24: {
        0: (-1, -3, 3, -1, 3, 1, 3, -1, 1, -3, -1, -3, -1, 1, 3, -3, -1, -3, 3, 3, 3, -3, -3, -3),
        1: (-1, -3, 3, 1, 1, -3, 1, -3, -3, 1, -3, -1, -1, 3, -3, 3, 3, 3, -3, 1, 3, 3, -3, -3),
        2: (-1, -3, -3, 1, -1, -1, -3, 1, 3, -1, -3, -1, -1, -3, 1, 1, 3, 1, -3, -1, -1, 3, -3, -3),
        3: (1, -3, 3, -1, -3, -1, 3, 3, 1, -1, 1, 1, 3, -3, -1, -3, -3, -3, -1, 3, -3, -1, -3, -3),
        4: (-1, 3, -3, -3, -1, 3, -1, -1, 1, 3, 1, 3, -1, -1, -3, 1, 3, 1, -1, -3, 1, -1, -3, -3),
        5: (-3, -1, 1, -3, -3, 1, 1, -3, 3, -1, -1, -3, 1, 3, 1, -1, -3, -1, -3, 1, -3, -3, -3, -3),
        6: (-3, 3, 1, 3, -1, 1, -3, 1, -3, 1, -1, -3, -1, -3, -3, -3, -3, -1, -1, -1, 1, 1, -3, -3),
        7: (-3, 1, 3, -1, 1, -1, 3, -3, 3, -1, -3, -1, -3, 3, -1, -1, -1, -3, -1, -1, -3, 3, 3, -3),
        8: (-3, 1, -3, 3, -1, -1, -1, -3, 3, 1, -1, -3, -1, 1, 3, -1, 1, -1, 1, -3, -3, -3, -3, -3),
        9: (1, 1, -1, -3, -1, 1, 1, -3, 1, -1, 1, -3, 3, -3, -3, 3, -1, -3, 1, 3, -3, 1, -3, -3),
        10: (-3, -3, -3, -1, 3, -3, 3, 1, 3, 1, -3, -1, -1, -3, 1, 1, 3, 1, -1, -3, 3, 1, 3, -3),
        11: (-3, 3, -1, 3, 1, -1, -1, -1, 3, 3, 1, 1, 1, 3, 3, 1, -3, -3, -1, 1, -3, 1, 3, -3),
        12: (3, -3, 3, -1, -3, 1, 3, 1, -1, -1, -3, -1, 3, -3, 3, -1, -1, 3, 3, -3, -3, 3, -3, -3),
        13: (-3, 3, -1, 3, -1, 3, 3, 1, 1, -3, 1, 3, -3, 3, -3, -3, -1, 1, 3, -3, -1, -1, -3, -3),
        14: (-3, 1, -3, -1, -1, 3, 1, 3, -3, 1, -1, 3, 3, -1, -3, 3, -3, -1, -1, -3, -3, -3, 3, -3),
        15: (-3, -1, -1, -3, 1, -3, -3, -1, -1, 3, -1, 1, -1, 3, 1, -3, -1, 3, 1, 1, -1, -1, -3, -3),
        16: (-3, -3, 1, -1, 3, 3, -3, -1, 1, -1, -1, 1, 1, -1, -1, 3, -3, 1, -3, 1, -1, -1, -1, -3),
        17: (3, -1, 3, -1, 1, -3, 1, 1, -3, -3, 3, -3, -1, -1, -1, -1, -1, -3, -3, -1, 1, 1, -3, -3),
        18: (-3, 1, -3, 1, -3, -3, 1, -3, 1, -3, -3, -3, -3, -3, 1, -3, -3, 1, 1, -3, 1, 1, -3, -3),
        19: (-3, -3, 3, 3, 1, -1, -1, -1, 1, -3, -1, 1, -1, 3, -3, -1, -3, -1, -1, 1, -3, 3, -1, -3),
        20: (-3, -3, -1, -1, -1, -3, 1, -1, -3, -1, 3, -3, 1, -3, 3, -3, 3, 3, 1, -1, -1, 1, -3, -3),
        21: (3, -1, 1, -1, 3, -3, 1, 1, 3, -1, -3, 3, 1, -3, 3, -1, -1, -1, -1, 1, -3, -3, -3, -3),
        22: (-3, 1, -3, 3, -3, 1, -3, 3, 1, -1, -3, -1, -3, -3, -3, -3, 1, 3, -1, 1, 3, 3, 3, -3),
        23: (-3, -1, 1, -3, -1, -1, 1, 1, 1, 3, 3, -1, 1, -1, 1, -1, -1, -3, -3, -3, 3, 1, -1, -3),
        24: (-3, 3, -1, -3, -1, -1, -1, 3, -1, -1, 3, -3, -1, 3, -3, 3, -3, -1, 3, 1, 1, -1, -3, -3),
        25: (-3, 1, -1, -3, -3, -1, 1, -3, -1, -3, 1, 1, -1, 1, 1, 3, 3, 3, -1, 1, -1, 1, -1, -3),
        26: (-1, 3, -1, -1, 3, 3, -1, -1, -1, 3, -1, -3, 1, 3, 1, 1, -3, -3, -3, -1, -3, -1, -3, -3),
        27: (3, -3, -3, -1, 3, 3, -3, -1, 3, 1, 1, 1, 3, -1, 3, -3, -1, 3, -1, 3, 1, -1, -3, -3),
        28: (-3, 1, -3, 1, -3, 1, 1, 3, 1, -3, -3, -1, 1, 3, -1, -3, 3, 1, -1, -3, -3, -3, -3, -3),
        29: (3, -3, -1, 1, 3, -1, -1, -3, -1, 3, -1, -3, -1, -3, 3, -1, 3, 1, 1, -3, 3, -3, -3, -3),
    },
}

SHORT_LOW_PAPR_TYPE2_PHASES_6: dict[int, tuple[int, ...]] = {
    0: (0, 0, 0, 0, 0, 0),
    1: (0, 1, 0, 3, 0, 5),
    2: (0, 2, 0, 2, 0, 2),
    3: (0, 3, 0, 1, 0, 7),
    4: (0, 4, 0, 0, 0, 4),
    5: (0, 5, 0, 7, 0, 3),
    6: (0, 6, 0, 6, 0, 6),
    7: (0, 7, 0, 5, 0, 1),
    8: (1, 0, 3, 0, 5, 0),
    9: (1, 1, 3, 3, 5, 5),
    10: (1, 2, 3, 2, 5, 2),
    11: (1, 3, 3, 1, 5, 7),
    12: (1, 4, 3, 0, 5, 4),
    13: (1, 5, 3, 7, 5, 3),
    14: (1, 6, 3, 6, 5, 6),
    15: (1, 7, 3, 5, 5, 1),
    16: (2, 0, 2, 0, 2, 0),
    17: (2, 1, 2, 3, 2, 5),
    18: (2, 2, 2, 2, 2, 2),
    19: (2, 3, 2, 1, 2, 7),
    20: (2, 4, 2, 0, 2, 4),
    21: (2, 5, 2, 7, 2, 3),
    22: (2, 6, 2, 6, 2, 6),
    23: (2, 7, 2, 5, 2, 1),
    24: (3, 0, 1, 0, 7, 0),
    25: (3, 1, 1, 3, 7, 5),
    26: (3, 2, 1, 2, 7, 2),
    27: (3, 3, 1, 1, 7, 7),
    28: (3, 4, 1, 0, 7, 4),
    29: (3, 5, 1, 7, 7, 3),
}

SHORT_LOW_PAPR_TYPE2_BITS: dict[int, dict[int, tuple[int, ...]]] = {
    12: {
        0: (0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0),
        1: (0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0),
        2: (0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0),
        3: (0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0),
        4: (0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0),
        5: (0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0),
        6: (0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0),
        7: (0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0),
        8: (0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0),
        9: (0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0),
        10: (1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0),
        11: (1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0),
        12: (1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0),
        13: (1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0),
        14: (1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0),
        15: (1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0),
        16: (1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0),
        17: (1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0),
        18: (1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0),
        19: (1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0),
        20: (1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0),
        21: (1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0),
        22: (1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0),
        23: (1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0),
        24: (1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0),
        25: (1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0),
        26: (0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0),
        27: (0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0),
        28: (0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0),
        29: (0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0),
    },
    18: {
        0: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0),
        1: (0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0),
        2: (0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0),
        3: (0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0),
        4: (0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0),
        5: (0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0),
        6: (0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0),
        7: (0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0),
        8: (0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0),
        9: (0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0),
        10: (1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0),
        11: (1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0),
        12: (1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0),
        13: (1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0),
        14: (1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0),
        15: (1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0),
        16: (1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0),
        17: (1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0),
        18: (1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0),
        19: (1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0),
        20: (1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0),
        21: (1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0),
        22: (1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0),
        23: (1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0),
        24: (1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0),
        25: (1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0),
        26: (0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0),
        27: (0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0),
        28: (0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0),
        29: (0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0),
    },
    24: {
        0: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0),
        1: (0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0),
        2: (0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0),
        3: (0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0),
        4: (0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0),
        5: (0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0),
        6: (0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0),
        7: (0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0),
        8: (0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0),
        9: (0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0),
        10: (1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0),
        11: (1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0),
        12: (1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0),
        13: (1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0),
        14: (1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0),
        15: (1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0),
        16: (1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0),
        17: (1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0),
        18: (1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0),
        19: (1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0),
        20: (1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0),
        21: (1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0),
        22: (1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0),
        23: (1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0),
        24: (1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0),
        25: (1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0),
        26: (0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0),
        27: (0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0),
        28: (0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0),
        29: (0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0),
    },
}


def gold_sequence(c_init: int, length: int) -> np.ndarray:
    """Generate the NR Gold sequence bits.

    Args:
        c_init: Scalar 31-bit sequence initialization value.
        length: Number of output bits.

    Returns:
        One-dimensional binary array with shape ``(length,)``; axis 0 is PRBS bit index.
    """
    nc = 1600
    seq_len = nc + length + 31
    x1 = np.zeros(seq_len, dtype=np.int8)
    x2 = np.zeros(seq_len, dtype=np.int8)
    x1[0] = 1

    for bit_idx in range(31):
        x2[bit_idx] = (c_init >> bit_idx) & 1

    for idx in range(31, seq_len):
        x1[idx] = (x1[idx - 28] + x1[idx - 31]) & 1
        x2[idx] = (x2[idx - 28] + x2[idx - 29] + x2[idx - 30] + x2[idx - 31]) & 1

    return (x1[nc : nc + length] + x2[nc : nc + length]) & 1


def qpsk_from_prbs(bits: np.ndarray) -> np.ndarray:
    """Map PRBS bits to unit-power QPSK symbols.

    Args:
        bits: One-dimensional binary array with shape ``(2 * num_symbols,)``; even
            indices drive the I component and odd indices drive the Q component.

    Returns:
        One-dimensional complex QPSK array with shape ``(num_symbols,)``.
    """
    real = 1 - 2 * bits[0::2]
    imag = 1 - 2 * bits[1::2]
    return (real + 1j * imag) / np.sqrt(2.0)


def pi_over_two_bpsk_from_bits(bits: np.ndarray) -> np.ndarray:
    """Map bits to pi/2-BPSK symbols.

    Args:
        bits: One-dimensional binary array with shape ``(num_symbols,)``; axis 0 is
            output symbol index before alternating pi/2 rotation.

    Returns:
        One-dimensional complex pi/2-BPSK sequence with shape ``(num_symbols,)``.
    """
    bits = np.asarray(bits, dtype=np.int8).reshape(-1)
    real = np.where(bits == 0, 1.0, -1.0)
    imag = real.copy()
    odd = np.arange(bits.size) % 2 == 1
    real[odd] *= -1.0
    return (real + 1j * imag) / np.sqrt(2.0)


def _largest_prime_less_than_or_equal(value: int) -> int:
    if value <= 2:
        return 2
    for candidate in range(value, 1, -1):
        is_prime = True
        limit = int(math.sqrt(candidate)) + 1
        for factor in range(2, limit):
            if candidate % factor == 0:
                is_prime = False
                break
        if is_prime:
            return candidate
    return 2


def _zadoff_chu_extension(root: int, length: int) -> np.ndarray:
    """Generate an extended Zadoff-Chu sequence.

    Args:
        root: Scalar Zadoff-Chu root index.
        length: Number of requested output symbols.

    Returns:
        One-dimensional complex sequence with shape ``(length,)``.
    """
    nzc = _largest_prime_less_than_or_equal(length)
    n = np.arange(nzc)
    base = np.exp(-1j * np.pi * root * n * (n + 1) / nzc)
    return base[np.mod(np.arange(length), nzc)]


@dataclass(frozen=True)
class DmrsInfo:
    symbol_indices: tuple[int, ...]
    re_offsets: np.ndarray
    re_per_prb: int


class DmrsGenerator(DmrsSequenceGenerator):
    """
    Protocol-oriented DMRS helper.

    Sequence initialization follows the standard Gold-sequence form used by
    TS 38.211 for PDSCH DM-RS and for PUSCH when transform precoding is
    disabled. For transform-precoded PUSCH, the implementation follows the
    low-PAPR sequence rules from clauses 5.2.2 and 5.2.3, including the
    short-sequence tables and the pi/2-BPSK DM-RS branch introduced by
    dmrs-UplinkTransformPrecoding-r16.
    """

    def get_dmrs_info(self, config: SimulationConfig) -> DmrsInfo:
        """Resolve DMRS positions inside one slot.

        Args:
            config: Full simulation configuration defining channel, waveform and DMRS type.

        Returns:
            DMRS metadata. ``re_offsets`` has shape ``(dmrs_re_per_prb,)``; axis 0
            enumerates RE offsets inside one PRB. ``symbol_indices`` enumerates OFDM
            symbol indices carrying DMRS.
        """
        if (
            config.link.channel_type.upper() == "PUSCH"
            and config.link.waveform.upper() == "DFT-S-OFDM"
        ):
            # For transform-precoded PUSCH DM-RS, clause 6.4.1.1.3 uses
            # k = 4n + 2k' + Δ with k' = 0,1, which gives 6 RE/PRB in the
            # current single-port implementation.
            re_offsets = np.array([0, 2, 4, 6, 8, 10], dtype=int)
        elif config.dmrs.config_type == 1:
            re_offsets = np.array([0, 2, 4, 6, 8, 10], dtype=int)
        elif config.dmrs.config_type == 2:
            re_offsets = np.array([0, 1, 6, 7], dtype=int)
        else:
            raise ValueError(f"Unsupported DMRS configuration type: {config.dmrs.config_type}")

        return DmrsInfo(
            symbol_indices=self._dmrs_symbol_indices(config),
            re_offsets=re_offsets,
            re_per_prb=re_offsets.size,
        )

    def generate_for_symbol(self, symbol: int, config: SimulationConfig) -> np.ndarray:
        """Generate DMRS symbols for one OFDM symbol.

        Args:
            symbol: Scalar OFDM symbol index inside the slot.
            config: Full simulation configuration defining sequence initialization.

        Returns:
            One-dimensional complex DMRS array with shape ``(num_dmrs_re_in_symbol,)``;
            axis 0 follows mapper RE order over the scheduled bandwidth.
        """
        info = self.get_dmrs_info(config)
        num_prbs = config.link.num_prbs
        if config.link.channel_type.upper() == "PUSCH" and config.link.waveform.upper() == "DFT-S-OFDM":
            return self._generate_pusch_transform_precoded(symbol, num_prbs, info, config)
        if config.link.channel_type.upper() == "PUSCH":
            return self._generate_gold_dmrs(symbol, num_prbs, info, config)
        if config.link.channel_type.upper() == "PDSCH":
            return self._generate_gold_dmrs(symbol, num_prbs, info, config)
        raise ValueError(f"Unsupported channel type: {config.link.channel_type}")

    def _dmrs_symbol_indices(self, config: SimulationConfig) -> tuple[int, ...]:
        """Resolve the OFDM symbol indices that carry DMRS.

        Args:
            config: Full simulation configuration with explicit or table-driven DMRS positions.

        Returns:
            Tuple of scalar OFDM symbol indices in ascending mapping order.
        """
        if config.dmrs.symbol_positions:
            return tuple(config.dmrs.symbol_positions)
        return resolve_dmrs_symbol_indices(
            start_symbol=int(config.link.start_symbol),
            num_symbols=int(config.link.num_symbols),
            mapping_type=str(config.dmrs.mapping_type),
            additional_positions=int(config.dmrs.additional_positions),
            max_length=int(config.dmrs.max_length),
            type_a_position=int(config.dmrs.type_a_position),
        )

    def _effective_scrambling_id(self, config: SimulationConfig) -> int:
        if config.dmrs.nid_nscid is not None:
            return int(config.dmrs.nid_nscid)
        if config.dmrs.scrambling_id0 is not None:
            return int(config.dmrs.scrambling_id0)
        return int(config.scrambling.n_id)

    def _dmrs_c_init(self, symbol: int, config: SimulationConfig) -> int:
        nid = self._effective_scrambling_id(config)
        slot = config.slot_index
        symbols_per_slot = config.carrier.symbols_per_slot
        return (
            (1 << 17) * (symbols_per_slot * slot + symbol + 1) * (2 * nid + 1)
            + 2 * nid
            + config.dmrs.n_scid
        ) % (1 << 31)

    def _generate_gold_dmrs(
        self,
        symbol: int,
        num_prbs: int,
        info: DmrsInfo,
        config: SimulationConfig,
    ) -> np.ndarray:
        """Generate CP-OFDM style Gold-sequence DMRS for one symbol.

        Args:
            symbol: Scalar OFDM symbol index inside the slot.
            num_prbs: Number of scheduled PRBs.
            info: DMRS metadata whose ``re_offsets`` axis enumerates REs per PRB.
            config: Full simulation configuration defining scrambling IDs.

        Returns:
            One-dimensional complex QPSK DMRS sequence with shape
            ``(num_prbs * dmrs_re_per_prb,)``.
        """
        dmrs_begin = info.re_per_prb * config.link.prb_start * 2
        dmrs_end = info.re_per_prb * (config.link.prb_start + num_prbs) * 2
        dmrs_size = info.re_per_prb * config.carrier.n_size_grid * 2
        bits = gold_sequence(self._dmrs_c_init(symbol, config), dmrs_size)
        return qpsk_from_prbs(bits[dmrs_begin:dmrs_end])

    def _generate_pusch_transform_precoded(
        self,
        symbol: int,
        num_prbs: int,
        info: DmrsInfo,
        config: SimulationConfig,
    ) -> np.ndarray:
        """Generate transform-precoded PUSCH low-PAPR DMRS for one symbol.

        Args:
            symbol: Scalar OFDM symbol index inside the slot.
            num_prbs: Number of scheduled PRBs.
            info: DMRS metadata whose ``re_offsets`` axis enumerates REs per PRB.
            config: Full simulation configuration defining hopping and pi/2-BPSK flags.

        Returns:
            One-dimensional constant-envelope DMRS sequence with shape
            ``(num_prbs * dmrs_re_per_prb,)``.
        """
        length = num_prbs * info.re_per_prb
        if self._use_type2_low_papr_sequence(config):
            return self._generate_type2_low_papr_sequence(symbol, length, config)

        u, v = self._pusch_low_papr_group_sequence_numbers(symbol, length, config)
        return self._low_papr_type1(u=u, v=v, length=length)

    def _pusch_low_papr_group_sequence_numbers(
        self,
        symbol: int,
        sequence_length: int,
        config: SimulationConfig,
    ) -> tuple[int, int]:
        """Resolve low-PAPR group and sequence hopping indices.

        Args:
            symbol: Scalar OFDM symbol index inside the slot.
            sequence_length: Number of DMRS symbols in the sequence.
            config: Full simulation configuration defining hopping options.

        Returns:
            Tuple ``(u, v)`` of scalar low-PAPR group and sequence numbers.
        """
        n_id_rs = config.dmrs.n_pusch_identity
        if n_id_rs is None:
            n_id_rs = self._effective_scrambling_id(config)

        slot_number = config.slot_index
        symbols_per_slot = config.carrier.symbols_per_slot
        linear_symbol = slot_number * symbols_per_slot + symbol

        f_gh = 0
        if config.dmrs.group_hopping:
            prbs = gold_sequence(c_init=int(n_id_rs // 30), length=8 * (linear_symbol + 1))
            hop_bits = prbs[8 * linear_symbol : 8 * (linear_symbol + 1)]
            f_gh = int(np.sum(hop_bits * (2 ** np.arange(8)))) % 30

        v = 0
        if config.dmrs.sequence_hopping and not config.dmrs.group_hopping and sequence_length >= 72:
            prbs = gold_sequence(c_init=int(n_id_rs // 30), length=linear_symbol + 1)
            v = int(prbs[linear_symbol])

        u = (f_gh + int(n_id_rs)) % 30
        return u, v

    def _low_papr_type1(self, u: int, v: int, length: int) -> np.ndarray:
        """Generate low-PAPR sequence type 1.

        Args:
            u: Scalar group number.
            v: Scalar sequence number.
            length: Number of requested DMRS symbols.

        Returns:
            One-dimensional complex sequence with shape ``(length,)``.
        """
        if length in SHORT_LOW_PAPR_TYPE1_PHASES:
            phases = np.array(SHORT_LOW_PAPR_TYPE1_PHASES[length][u], dtype=np.float64)
            sequence = np.exp(1j * np.pi * phases / 4.0)
            return sequence / np.sqrt(np.mean(np.abs(sequence) ** 2))
        return self._zc_low_papr_sequence(u=u, v=v, length=length)

    def _generate_type2_low_papr_sequence(
        self,
        symbol: int,
        length: int,
        config: SimulationConfig,
    ) -> np.ndarray:
        """Generate low-PAPR sequence type 2 for pi/2-BPSK transform precoding.

        Args:
            symbol: Scalar OFDM symbol index inside the slot.
            length: Number of requested DMRS symbols.
            config: Full simulation configuration defining scrambling IDs.

        Returns:
            One-dimensional complex sequence with shape ``(length,)``.
        """
        if length == 6:
            phases = np.array(SHORT_LOW_PAPR_TYPE2_PHASES_6[self._type2_u_index(symbol, config)], dtype=np.float64)
            return np.exp(1j * np.pi * phases / 4.0)

        if length in SHORT_LOW_PAPR_TYPE2_BITS:
            bits = np.array(SHORT_LOW_PAPR_TYPE2_BITS[length][self._type2_u_index(symbol, config)], dtype=np.int8)
            return pi_over_two_bpsk_from_bits(bits)

        bits = gold_sequence(self._pi2_bpsk_dmrs_c_init(symbol, config), length)
        return pi_over_two_bpsk_from_bits(bits)

    def _type2_u_index(self, symbol: int, config: SimulationConfig) -> int:
        n_id = self._effective_pi2_bpsk_scrambling_id(config)
        slot = config.slot_index
        symbols_per_slot = config.carrier.symbols_per_slot
        linear_symbol = slot * symbols_per_slot + symbol
        if config.dmrs.group_hopping:
            prbs = gold_sequence(c_init=int(n_id // 30), length=8 * (linear_symbol + 1))
            hop_bits = prbs[8 * linear_symbol : 8 * (linear_symbol + 1)]
            f_gh = int(np.sum(hop_bits * (2 ** np.arange(8)))) % 30
        else:
            f_gh = 0
        return (f_gh + int(n_id)) % 30

    def _effective_pi2_bpsk_scrambling_id(self, config: SimulationConfig) -> int:
        if config.dmrs.n_scid == 0 and config.dmrs.pi2bpsk_scrambling_id0 is not None:
            return int(config.dmrs.pi2bpsk_scrambling_id0)
        if config.dmrs.n_scid == 1 and config.dmrs.pi2bpsk_scrambling_id1 is not None:
            return int(config.dmrs.pi2bpsk_scrambling_id1)
        return self._effective_scrambling_id(config)

    def _pi2_bpsk_dmrs_c_init(self, symbol: int, config: SimulationConfig) -> int:
        nid = self._effective_pi2_bpsk_scrambling_id(config)
        slot = config.slot_index
        symbols_per_slot = config.carrier.symbols_per_slot
        return (
            (1 << 17) * (symbols_per_slot * slot + symbol + 1) * (2 * nid + 1)
            + 2 * nid
            + config.dmrs.n_scid
        ) % (1 << 31)

    @staticmethod
    def _use_type2_low_papr_sequence(config: SimulationConfig) -> bool:
        return (
            config.link.channel_type.upper() == "PUSCH"
            and config.link.waveform.upper() == "DFT-S-OFDM"
            and config.link.modulation.upper() == "PI/2-BPSK"
            and bool(config.dmrs.uplink_transform_precoding)
        )

    @staticmethod
    def _zc_low_papr_sequence(u: int, v: int, length: int) -> np.ndarray:
        """Generate long low-PAPR sequence from a Zadoff-Chu root.

        Args:
            u: Scalar group number.
            v: Scalar sequence number.
            length: Number of requested DMRS symbols.

        Returns:
            One-dimensional normalized complex sequence with shape ``(length,)``.
        """
        nzc = _largest_prime_less_than_or_equal(length)
        q_bar = nzc * (u + 1) / 31.0
        q = int(np.floor(q_bar + 0.5)) + v * ((-1) ** int(np.floor(2 * q_bar)))
        n = np.arange(nzc)
        base = np.exp(-1j * np.pi * q * n * (n + 1) / nzc)
        sequence = base[np.mod(np.arange(length), nzc)]
        return sequence / np.sqrt(np.mean(np.abs(sequence) ** 2))
