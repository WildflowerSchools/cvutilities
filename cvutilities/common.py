import numpy as np


BODY_PART_NAMES = (
    "Nose",
    "Neck",
    "RShoulder",
    "RElbow",
    "RWrist",
    "LShoulder",
    "LElbow",
    "LWrist",
    "RHip",
    "RKnee",
    "RAnkle",
    "LHip",
    "LKnee",
    "LAnkle",
    "REye",
    "LEye",
    "REar",
    "LEar"
)

NUM_BODY_PARTS = len(BODY_PART_NAMES)

BODY_PART_CONNECTORS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (1, 5),
    (5, 6),
    (6, 7),
    (1, 8),
    (8, 9),
    (9, 10),
    (1, 11),
    (11, 12),
    (12, 13),
    (0, 14),
    (14, 16),
    (0, 15),
    (15, 17)
)

# Define some subsets of indices that we will use when calculating anchor points
NECK_INDEX = 1
SHOULDER_INDICES = (2, 5)
HEAD_AND_TORSO_INDICES = (0, 1, 2, 5, 8, 11, 14 , 15, 16, 17)

# Specify time unit when unitless time value is needed (e.g., in Kalman filter)
TIME_UNIT = np.timedelta64(1, 's')
