import numpy as np
from casadi import vertcat, sin, cos, tan
import pdb

def single_integrator(x, u, params=None):
    """ x = [x], u = [v_x] 
    Dynamics: xdot = u
    """
    return u[0]

def single_integrator_2D(x, u, params=None):
    """ x = [x, y], u = [v_x, v_y] 
    Dynamics: xdot = u
    """
    return vertcat(
        u[0],
        u[1],
        )

def single_integrator_3D(x, u, params=None):
    """ x = [x, y, z], u = [v_x, v_y, v_z] 
    Dynamics: xdot = u
    """
    return vertcat(
        u[0],
        u[1],
        u[2],
        )

def double_integrator(x, u, params=None):
    """ x = [x, y, xdot, ydot], u = [a_x, a_y] 
    Dynamcis: xddot = u
    """
    return vertcat(
        x[2],
        x[3],
        u[0],
        u[1],
        )

def quadcopter(x, u, params=None):
    """ x = [x, y, z, phi, th, psi, xd, yd, zd, p, q, r]
        u = [T, Mx, My, Mz], 
        m is vehicle mass and I is principle axis MOI """
    m = params["m"]
    I = params["I"]
    return vertcat(
        x[6], 
        x[7], 
        x[8],
        x[9] + (x[10]*sin(x[3]) + x[11]*cos(x[3]))*tan(x[4]),
        x[10]*cos(x[3]) - x[11]*sin(x[3]),
        (x[10]*sin(x[3]) + x[11]*cos(x[3]))*(1.0/cos(x[4])),
        (u[0]/m)*(sin(x[3])*sin(x[5]) + cos(x[3])*sin(x[4])*cos(x[5])),
        (u[0]/m)*(cos(x[3])*sin(x[4])*sin(x[5]) - sin(x[3])*cos(x[5])),
        (u[0]/m)*(cos(x[3])*cos(x[4])) - 9.81,
        (1.0/I[0,0])*(u[1] - (I[2,2]-I[1,1])*x[10]*x[11]),
        (1.0/I[1,1])*(u[2] - (I[0,0]-I[2,2])*x[11]*x[9]),
        (1.0/I[2,2])*(u[3] - (I[1,1]-I[0,0])*x[9]*x[10]),    
        )

def van_der_pol(x, u, params=None):
    """ x = [x0, x1] """
    return vertcat(
        (1 - x[1]**2)*x[0] - x[1] + u,
        x[0],
        )

def gnss_pos_and_bias(x, u, params=None):
    """ x = [x, y, z, b, bd] where b is receiver clock
    bias that is assumed to be linearly changing
    Dynamics: xdot = u and bdot = bd
    """
    return vertcat(
        u[0],
        u[1],
        u[2],
        x[4],
        0.0,
        )