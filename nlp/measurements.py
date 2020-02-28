import numpy as np
from casadi import vertcat, sin, cos, tan, sqrt


def full_state(x, params=None):
    return x


def pseudorange(x, params=None):
    """ x = [x, y, z, b] where b is the receiver
    clock bias term """
    return sqrt((x[0] - params["sat_pos"][0])**2 +
                (x[1] - params["sat_pos"][1])**2 +
                (x[2] - params["sat_pos"][2])**2) + \
                x[3]