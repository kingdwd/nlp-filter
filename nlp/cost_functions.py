import numpy as np

def van_der_pol(x, u, params=None):
    """ x = [x0, x1] """
    return x[0]**2 + x[1]**2 + u**2

def single_integrator(x, u, params=None):
    """ x = [x0, x1], u = [u0, u1] """
    return x[0]**2 + x[1]**2 + u[0]**2 + u[1]**2
