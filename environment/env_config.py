import numpy as np

RRT_EPS = 0.05  #2D
STICK_LENGTH = 1.5 * 2 / 15
LIMITS = np.array([1., 1., 8.*RRT_EPS])