import numpy as np
from casadi import vertcat, sin, cos, tan

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

def multi_receiver(x, params=None):
    """ x = [xB, yB, zB, bB, xdB, ydB, zdB, alphaB] 
    where b is receiver clock bias that is assumed to be linearly 
    changing at rate alpha/
    Dynamics: xdot = xd and bdot = alpha
    """
    return vertcat(
        x[4],
        x[5],
        x[6],
        x[7],
        0.0,
        0.0,
        0.0,
        0.0,
        )

def gnss_two_receiver(x, u, params=None):
    """ x = [xA, yA, zA, bA, alphaA, xB, yB, zB, bB, alphaB]
    where b is receiver clock bias that is assumed to be linearly 
    changing at rate alpha
    Dynamics: xdot = xd and bdot = alpha
    """
    return vertcat(
        u[0],
        u[1],
        u[2],
        x[4],
        0.0,
        u[3],
        u[4],
        u[5],
        x[9],
        0.0,
        )

def kinematic_bycicle_and_bias(x, u, params=None):
    """ kinematic bycicle model: x = [x, y, z, b, bd, th], u = [throttle, steer]
        xd = vcos(th)
        yd = vsin(th)
        zd = 0.0
        bd = bd
        bdd = 0.0
        thd = v/L tan(delta)""" 
    L = 0.28 # meters
    v= 8.72649116358*u[0] - 0.856053299155
    delta = np.deg2rad(28)*u[1]

    return vertcat(
        v*np.cos(x[2]),
        v*np.sin(x[2]),
        0.0,
        x[4],
        0.0,
        (v/L)*np.tan(delta),
        )

def linear_tire_model(F_xr, delta, vx, vy, r):
    a_r = (vy - D_R*r)/vx
    a_f = (vy + D_F*r)/vx - delta
    C_a = [C_AR, C_AF]
    a = [a_r, a_f] # [alpha_r, alpha_f]
    F_y = np.zeros(2)
    for i in range(2):
        F_y[i] = -C_a[i]*a[i]
    return F_y

def vehicle_dynamics(x, u, params=None):
    """ x = [px, py, psi, vx, vy, r] 
        u = [F_xr, delta] = [rear axle thrust, steering angle] """
    C = params["car_params"]

    epsilon = .001
    F_yr = -C["C_AR"]*(x[4] - C["D_R"]*x[5])/(x[3] + epsilon)
    F_yf = -C["C_AF"]*((x[4] + C["D_F"]*x[5])/(x[3] + epsilon) - u[1])

    return vertcat(
        x[3]*cos(x[2]) - x[4]*sin(x[2]),
        x[3]*sin(x[2]) + x[4]*cos(x[2]), 
        x[5], 
        (-F_yf*sin(u[1]) + u[0])/C["M"] + x[5]*x[4], 
        (F_yf*cos(u[1]) + F_yr)/C["M"] - x[5]*x[3], 
        (C["D_F"]*F_yf*cos(u[1]) - C["D_R"]*F_yr)/C["I_Z"],
        )

def vehicle_dynamics_and_gnss(x, u, params=None):
    """ [px, py, psi, vx, vy, psid, b, bd, pz] """
    xd_vehicle = vehicle_dynamics(x[:6], u, params)
    return vertcat(
        xd_vehicle,
        x[7],
        0.0,
        0.0,
        )