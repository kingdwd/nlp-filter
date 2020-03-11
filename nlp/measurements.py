import numpy as np
from casadi import vertcat, sin, cos, tan, sqrt, dot, norm_2

def full_state(x, params=None):
    return x

def multi_receiver_range(x, params=None):
    """ x = [x, y, z, b, xd, yd, zd, alpha] """
    y = params["y"]
    return sqrt((x[0] - y[0])**2 +
                (x[1] - y[1])**2 +
                (x[2] - y[2])**2 + .00001)

def pseudorange(x, params=None):
    """ x = [x, y, z, b, ...] where b is the receiver
    clock bias term, assumes sat pos and x are in same
    coordinate representation.
        y = [pseudorange] """
    return sqrt((x[0] - params["sat_pos"][0])**2 +
                (x[1] - params["sat_pos"][1])**2 +
                (x[2] - params["sat_pos"][2])**2) + \
                x[3]

def pseudorange_rate(x, params=None):
    """ x = [x, y, z, b, xd, yd, zd, alpha, ...] where alpha is the receiver
    clock bias rate term, assumes sat vel and xd are in same
    coordinate representation.
        y = [pseudorange rate] """
    r = params["sat_pos"] - x[:3]
    LoS = r/norm_2(r)
    return dot(params["sat_vel"] - x[4:7], LoS) + x[7]

def vehicle_pseudorange(x, params=None):
    """ x = [px, py, psi, vx, vy, psid, b, bd, pz] where b is the receiver
    clock bias term 
        y = [pseudorange] """
    return sqrt((x[0] - params["sat_pos"][0])**2 +
                (x[1] - params["sat_pos"][1])**2 +
                (x[8] - params["sat_pos"][2])**2) + \
                x[6]