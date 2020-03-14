import numpy as np
import os
from matplotlib import pyplot as plt
import simulate
import data
import utils
import leastsquares
import data

C_AF = 1.1441e5 # N/rad
C_AR = 1.3388e5 # N/rad
MU = 0.75
M = 2009 # kg
D_F = 1.53 # m
D_R = 1.23 # m
I_Z = 2000 # kg m^2
H = 0.25 # m
G = 9.81 # m/s

def get_parameters():
    params = {"C_AF":C_AF, "C_AR":C_AR, "MU":MU, 
                "M":M, "D_F":D_F, "D_R":D_R, "I_Z":I_Z, "H":H, "G":G}
    return params

def tire_model(F_xr, delta, vx, vy, r):
    F_zr = (M*G*D_F + H*F_xr)/(D_F + D_R) # approximation
    F_zf = (M*G*D_R - H*F_xr)/(D_F + D_R) # approximation
    a_r = np.arctan2(vy - D_R*r, vx)
    a_f = np.arctan2(vy + D_F*r, vx) - delta

    C_a = [C_AR, C_AF]
    F_x = [F_xr, 0.0] # [F_xr, F_xf]
    F_z = [F_zr, F_zf]
    a = [a_r, a_f] # [alpha_r, alpha_f]
    F_y = np.zeros(2)
    for i in range(2):
        F_y_max_i = np.sqrt(MU**2*F_z[i]**2 - F_x[i]**2)
        gamma = np.abs((C_a[i]*np.tan(a[i]))/(3*F_y_max_i))

        if F_x[i] > MU*F_z[i]:
            F_y[i] = 0
        elif gamma < 1:
            F_y[i] = -C_a[i]*np.tan(a[i])*(1 - gamma + (1./3)*gamma**2)
        else:
            F_y[i] = -F_y_max_i*np.sign(np.tan(a[i]))
    return F_y


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

    tire_model_function = params["tire_model_func"]
    F_y = tire_model_function(u[0], u[1], x[3], x[4], x[5])

    pxd = x[3]*np.cos(x[2]) - x[4]*np.sin(x[2])
    pyd = x[3]*np.sin(x[2]) + x[4]*np.cos(x[2])
    psid = x[5]
    ax = (-F_y[1]*np.sin(u[1]) + u[0])/M + x[5]*x[4]
    ay = (F_y[1]*np.cos(u[1]) + F_y[0])/M - x[5]*x[3]
    rd = (D_F*F_y[1]*np.cos(u[1]) - D_R*F_y[0])/I_Z
    return np.array([pxd, pyd, psid, ax, ay, rd])

def GNSS_measurement_sim(t, p_ECEF, sat_pos, R, alpha, b0):
    """ Simulates a GNSS pseudorange measurement """
    e = np.sqrt(R)*np.random.randn(1)
    return np.linalg.norm(p_ECEF - sat_pos) + b0 + alpha*t + e


def compass_measurement_sim(heading, R):
    """ heading in radians """
    e = np.sqrt(R)*np.random.randn(1)
    return heading + e


def gyro_measurement_sim(yaw_rate, R):
    """ yaw rate in rad/s """
    e = np.sqrt(R)*np.random.randn(1)
    return yaw_rate + e


if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    data_path = os.path.join(dir_path, 'data')
    car_path = os.path.join(data_path, 'autonomous-car')

    # Simulate open loop sequence
    sim_def = data.load_obj(car_path + '/sim/sim_definition')
    x0 = sim_def["x0"]
    u = sim_def["u"]
    T = sim_def["T"]
    N = sim_def["N"]
    t = sim_def["t"]
    params = {"tire_model_func":tire_model}
    x = simulate.open_loop_sim(t, u, x0, vehicle_dynamics, params)

    # Reference location in ECEF
    lat0 = 37.4276
    lon0 = -122.1670
    h0 = 0
    p_ref_ECEF = utils.lla2ecef(np.array([lat0, lon0, h0]))

    # Load some data to get satellite positions
    data_gnss = data.load_gnss_logs(dir_path + '/data/rc-car/gnss/gnss_log_2020_02_27_10_02_20')
    data_sat_pos = data_gnss["sat_pos"][0]
    data_sat_vel = data_gnss["sat_vel"][0]

    # Compute car positions in ECEF and then extract a GPS pseudorange measurement
    R = 10
    alpha = 200 # m/s
    b0 = 0 # m
    N_sats = data_sat_pos.shape[0]
    t_gnss = []
    pr = []
    sat_pos = []
    for k in range(N+1):
        if k % 10 == 0:
            p_ENU_k = np.hstack((x[:2, k], np.zeros(1)))
            pos_ECEF_k = utils.enu2ecef(p_ENU_k, p_ref_ECEF)
            pr_k = np.zeros(N_sats)
            for i in range(N_sats):
                pr_k[i] = GNSS_measurement_sim(t[k], pos_ECEF_k, data_sat_pos[i,:], R, alpha, b0)
            pr.append(pr_k)
            sat_pos.append(data_sat_pos)
            t_gnss.append(t[k])
    gnss = {"t":np.array(t_gnss), "sat_pos":sat_pos, "pr":pr, "R":R, "alpha":alpha, "b0":b0}

    # Simulate compass measurements for heading
    heading = np.zeros(N+1)
    R_compass = np.deg2rad(5)
    psi = (x[2,:] + np.pi) % (2*np.pi) - np.pi # move to interval [-pi pi]
    for k in range(N+1):
        heading[k] = compass_measurement_sim(psi[k], R_compass)
    compass = {"t":t, "psi":heading, "R":R_compass}

    # Simulate gyro yaw rate measurements
    yaw_rate = np.zeros(N+1)
    R_yawrate = np.deg2rad(0.05)
    for k in range(N+1):
        yaw_rate[k] = gyro_measurement_sim(x[5,k], R_yawrate)
    gyro = {"t":t, "psid":yaw_rate, "R":R_yawrate}

    # Save data
    sensors = {"gnss":gnss, "compass":compass, "gyro":gyro}
    trajectory = {"t":t, "x":x, "u":u, "x0":x0, "dt":sim_def["dt"]}

    data.save_obj(sensors, car_path + '/sim/sensor_data')
    data.save_obj(trajectory, car_path + '/sim/traj_data')