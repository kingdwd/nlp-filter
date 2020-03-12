import numpy as np
from casadi import vertcat, sin, cos, tan, sqrt, dot, norm_2

def full_state(x, params=None):
    return x

def multi_receiver_range_2d(x, params=None):
    if "y" in params:
        if "idx" not in params:
            idx = [0, 1]
        else:
            idx = params["idx"]
        y = sqrt((x[idx[0]] - params["y"][0])**2 +
                    (x[idx[1]] - params["y"][1])**2 + .000001)
    elif "idxA" in params and "idxB" in params:
        idxA = params["idxA"]
        idxB = params["idxB"]
        y = sqrt((x[idxA[0]] - x[idxB[0]])**2 +
                    (x[idxA[1]] - x[idxB[1]])**2 + .000001)
    return y

def multi_receiver_range_3d(x, params=None):
    if "y" in params:
        if "idx" not in params:
            idx = [0, 1, 2]
        else:
            idx = params["idx"]
        y = sqrt((x[idx[0]] - params["y"][0])**2 +
                    (x[idx[1]] - params["y"][1])**2 +
                    (x[idx[2]] - params["y"][2])**2 + .000001)
    elif "idxA" in params and "idxB" in params:
        idxA = params["idxA"]
        idxB = params["idxB"]
        y = sqrt((x[idxA[0]] - x[idxB[0]])**2 +
                    (x[idxA[1]] - x[idxB[1]])**2 +
                    (x[idxA[2]] - x[idxB[2]])**2 + .000001)
    return y

def pseudorange(x, params=None):
    """ x = [x, y, z, b, ...] where b is the receiver
    clock bias term, assumes sat pos and x are in same
    coordinate representation. params["idx"] = [..] is
    a list of indices for [x,y,z,b] respectively in the
    vector x.
        y = [pseudorange] """
    if "idx" in params:
        idx = params["idx"]
    else:
        idx = [0, 1, 2, 3]
    return sqrt((x[idx[0]] - params["sat_pos"][0])**2 +
                (x[idx[1]] - params["sat_pos"][1])**2 +
                (x[idx[2]] - params["sat_pos"][2])**2) + \
                x[idx[3]]

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