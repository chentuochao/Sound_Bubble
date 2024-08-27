"""
A collection of constants that probably should not be changed
"""

# Universal Constants
SPEED_OF_SOUND = 343.0  # m/s

# Project constants
EPSILON = 1e-3
DIMENSIONS = 2
NORM_ORDER = 2
MAX_MIC_POSITION_ERROR = 0.02
MAX_RETRIES=3
SPOT_RADIUS = 0.1
MAX_SHIFT = 2

# ALL_WINDOW_SIZES = [
#     0.1,
#     0.3,
#     0.8,
#     1.3,
#     1.7
# ]

# ALL_WINDOW_SIZES = [
#     0.1,
#     0.3,
#     0.8
# ]

ALL_WINDOW_SIZES = [
    0.1
]


import numpy as np

# ALL_SIDE_LENGTHS = [
#     0.25,
#     0.5,
#     1,
#     2,
#     4,
#     8
# ]

ALL_SIDE_LENGTHS = [
    0.25,
    0.5,
    1
]

# ALL_SIDE_LENGTHS = [
#     0.25
# ]

ALL_SIDE_LENGTHS = np.array(ALL_SIDE_LENGTHS)

ALL_RADIUS_SIZES = ALL_SIDE_LENGTHS / 2 ** 0.5


