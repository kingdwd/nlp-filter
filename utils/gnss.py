import numpy as np


def pseudorange(x, params=None, jac=False):
    """ x = [x, y, z, b, bd] where b is receiver clock
    bias that is assumed to be linearly changing at rate
    bd """
    y = np.sqrt((x[0] - params["sat_pos"][0])**2 +
                    (x[1] - params["sat_pos"][1])**2 +
                    (x[2] - params["sat_pos"][2])**2) + \
                    x[3]
    if jac:
        J = np.zeros(5)

        # Line of sight vector
        LoS = params["sat_pos"] - x[:3]
        J[:3] = -LoS/np.linalg.norm(LoS)
        J[3] = 1.0
        return y, J
    else:
        return y


def multi_pseudorange(x, params=None, jac=False):
    """ x = [x, y, z, b, bd] where b is receiver clock
    bias that is assumed to be linearly changing at rate
    bd.
    Output is an array of psuedoranges """

    N = params["sat_pos"].shape[0]
    y = np.zeros(N)
    if jac:
        J = np.zeros((N,5))

    for i in range(N):
        if jac:
            y[i], J[i,:] = pseudorange(x, params={"sat_pos": params["sat_pos"][i,:]}, jac=True)
        else:
            y[i] = pseudorange(x, params={"sat_pos": params["sat_pos"][i,:]})

    if jac:
        return y, J
    else:
        return y


def multi_pseudorange_and_bias(x, params=None, jac=False):
    """ x = [x, y, z, b, bd] where b is receiver clock
    bias that is assumed to be linearly changing at rate
    bd.
    Output is an array of psuedoranges and bias """
    N = params["sat_pos"].shape[0]
    y = np.zeros(N+1)
    J = np.zeros((N+1, 5))
    y[-1] = x[3] # bias
    if jac:
        y[:-1], J[:-1,:] = multi_pseudorange(x, params={"sat_pos": params["sat_pos"]}, jac=True)
        return y, J
    else:
        y[:-1], J[:-1,:] = multi_pseudorange(x, params={"sat_pos": params["sat_pos"]}, jac=False)
        return y


def gnss_pos_and_bias(x, u, params=None, jac=False):
    """ x = [x, y, z, b, bd] where b is receiver clock
    bias that is assumed to be linearly changing
    Dynamics: x+ = x + dt*u and b+ = b + dt*bd
    """
    x += params["dt"]*np.array([u[0], u[1], u[2], x[4], 0.0])

    if jac:
        J = np.eye(5)
        J[3, 4] = params["dt"]
        return x, J
    else:
        return x




        
        