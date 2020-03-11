import numpy as np
import casadi


def van_der_pol(x, u, params=None):
    """ x = [x0, x1] """
    return x[0]**2 + x[1]**2 + u**2


def single_integrator(x, u, params=None):
    """ x = [x0, x1], u = [u0, u1] """
    return x[0]**2 + x[1]**2 + u[0]**2 + u[1]**2


def l2_norm(x, params=None):
    """ Returns the l2 norm ||x||^2 """
    return casadi.mtimes(x.T, x)


def weighted_l2_norm(x, params=None):
    """ Returns the weighted l2 norm ||x||_Q^2 """
    return casadi.mtimes(x.T, casadi.mtimes(params["Q"], x))


def pseudo_huber_loss(x, params=None):
    """ Returns the pseudo-huber loss using parameter delta and Q """
    n = params["Q"].shape[0]
    huber = 0
    for i in range(n):
        huber += 2*params["Q"][i,i]*params["delta"]**2*(casadi.sqrt(1 + (x[i])**2/params["delta"]**2) - 1.0)
    return huber