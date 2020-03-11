import numpy as np
import os
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import nlp.nlp as nlp
import nlp.dynamics as dynamics
import nlp.cost_functions as cost_functions
import nlp.constraints as constraints
import nlp.simulate as simulate
import nlp.measurements as measurements
import utils.gnss as gnss
import utils.utils as utils
import utils.ekf as ekf
import utils.data as data_utils
import utils.leastsquares as ls
import utils.vehicle_sim as vehicle_sim

def discrete_vehicle_dynamics(x, u, params=None, jac=False):
    """ Discretizing using explicit Euler
    x = [px, py, psi, vx, vy, psid, b, bd, pz]
    u = [F, steering] """
    dt = params["dt"]
    xd = vehicle_sim.vehicle_dynamics(x[:6], u, {"tire_model_func":vehicle_sim.linear_tire_model})
    xd = np.hstack((xd, np.array([x[7], 0.0, 0.0])))
    x += dt*xd

    C = params["car_params"] # constants
    if jac:
        J = np.eye(9)
        dF_yf_dvx = C["C_AF"]*(x[4] + C["D_F"]*x[5])*(1./x[3]**2)
        dF_yf_dvy = -C["C_AF"]/x[3]
        dF_yf_dr = -C["C_AF"]*C["D_F"]/x[3]
        dF_yr_dvx = C["C_AR"]*(x[4] - C["D_R"]*x[5])*(1./x[3]**2)
        dF_yr_dvy = -C["C_AR"]/x[3]
        dF_yr_dr = C["C_AR"]*C["D_R"]/x[3]
        J[0, 2] += params["dt"]*(-x[3]*np.sin(x[2]) - x[4]*np.cos(x[2]))
        J[0, 3] += params["dt"]*np.cos(x[2])
        J[0, 4] += -params["dt"]*np.sin(x[2])
        J[1, 2] += params["dt"]*(x[3]*np.cos(x[2]) - x[4]*np.sin(x[2]))
        J[1, 3] += params["dt"]*np.sin(x[2])
        J[1, 4] += params["dt"]*np.cos(x[2])
        J[2, 5] += params["dt"]
        J[3, 3] += -(params["dt"]/C["M"])*(np.sin(u[1])*dF_yf_dvx)
        J[3, 4] += params["dt"]*(x[5] - (np.sin(u[1])*dF_yf_dvy)/C["M"])
        J[3, 5] += params["dt"]*(x[4] - (np.sin(u[1])*dF_yf_dr)/C["M"])
        J[4, 3] += params["dt"]*((np.cos(u[1])*dF_yf_dvx + dF_yr_dvx)/C["M"] - x[5])
        J[4, 4] += (params["dt"]/C["M"])*(np.cos(u[1])*dF_yf_dvy + dF_yr_dvy)
        J[4, 5] += params["dt"]*((np.cos(u[1])*dF_yf_dr + dF_yr_dr)/C["M"] - x[3])
        J[5, 3] += (params["dt"]/C["I_Z"])*(C["D_F"]*np.cos(u[1])*dF_yf_dvx - C["D_R"]*dF_yr_dvx)
        J[5, 4] += (params["dt"]/C["I_Z"])*(C["D_F"]*np.cos(u[1])*dF_yf_dvy - C["D_R"]*dF_yr_dvy)
        J[5, 5] += (params["dt"]/C["I_Z"])*(C["D_F"]*np.cos(u[1])*dF_yf_dr - C["D_R"]*dF_yr_dr)
        J[6, 7] += params["dt"]
        return x, J
    else:
        return x

def vehicle_sensors_model(x, params=None, jac=False):
    # xmeas = [x,y,z,b,bd]
    x_meas = np.array([x[0], x[1], x[8], x[6], x[7]])
    if jac:
        y = np.array([])
        J = np.array([]).reshape(-1, 9)
        y_pr, J_pr = gnss.multi_pseudorange(x_meas, params=params, jac=True)
        N_pr = y_pr.shape[0] # Number of pseudorange measurements

        # Reorder to make sure the Jacobians match
        J_pr = np.hstack((J_pr[:,0].reshape(N_pr, 1), J_pr[:,1].reshape(N_pr, 1), 
                        np.zeros((N_pr, 4)), J_pr[:,3].reshape(N_pr, 1), J_pr[:,4].reshape(N_pr, 1), 
                        J_pr[:,2].reshape(N_pr, 1)))


        # Now combine into measurement and Jacobian
        y = np.hstack((y, y_pr))
        J = np.vstack((J, J_pr))

        return y, J
    else:
        y = gnss.multi_pseudorange(x_meas, params=params, jac=False)
        return y

def check_for_divergence(x, x_true):
    if np.linalg.norm(x-x_true) > 100:
        return True
    else:
        return False

# Set directory paths
dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(dir_path, 'data')
car_path = os.path.join(data_path, 'autonomous-car')

# Reference location on Earth (Hoover Tower)
lat0 = 37.4276
lon0 = -122.1670
h0 = 0
p_ref_ECEF = utils.lla2ecef(np.array([lat0, lon0, h0]))

# Load data
sensors = data_utils.load_obj(car_path + '/sim/sensor_data')
data_gnss = sensors["gnss"]
data_compass = sensors["compass"]
car_params = vehicle_sim.get_parameters()
traj = data_utils.load_obj(car_path + '/sim/traj_data')

################################################################
# Least Squares
################################################################
LS = ls.runLeastSquares(data_gnss["t"], data_gnss["sat_pos"], data_gnss["pr"], p_ref_ECEF=p_ref_ECEF)
data_utils.save_obj(LS, car_path + '/filtering/leastsquares')

################################################################
# Cost matrices for EKF and NLP
################################################################
Q_NLP = np.diag([0.01, 0.01, 0.01, 100, 500, 500, .001, .001, .001]) # covariance for dynamics
Q_EKF = .001*Q_NLP # 1/1000 factor due to time discretization of .01 and sampling time of .1
r_pr = float(data_gnss["R"]) # covariance for pseudorange measurement
P_NLP = np.diag(1*np.ones(9))
P_EKF = P_NLP

################################################################
# EKF
################################################################
dt = traj["dt"]
N_steps = traj["t"].shape[0]
EKF = {"t":traj["t"], "p_ref_ECEF":p_ref_ECEF, "bias":np.zeros(N_steps),
      "x_ENU":np.zeros(N_steps), "y_ENU":np.zeros(N_steps), "z_ENU":np.zeros(N_steps),
      "lat":np.zeros(N_steps), "lon":np.zeros(N_steps), "h":np.zeros(N_steps),
      "psi":np.zeros(N_steps), "vx":np.zeros(N_steps), "vy":np.zeros(N_steps)}

# Create EKF object
xhat0 = np.hstack((traj["x0"], np.array([data_gnss["b0"], data_gnss["alpha"], 0.0]))) 

ekf_filter = ekf.EKF(discrete_vehicle_dynamics, vehicle_sensors_model, xhat0, P_EKF)

# Run EKF
for (k, t) in enumerate(traj["t"]):
    EKF["x_ENU"][k] = ekf_filter.mu[0]
    EKF["y_ENU"][k] = ekf_filter.mu[1]
    EKF["z_ENU"][k] = ekf_filter.mu[8]
    EKF["psi"][k] = ekf_filter.mu[2]
    EKF["vx"][k] = ekf_filter.mu[3]
    EKF["vy"][k] = ekf_filter.mu[4]
    EKF["bias"][k] = ekf_filter.mu[6]

    # Only perform measurement correction when GNSS data is available
    i0 = np.where(data_gnss["t"] <= t + .00001)[0][-1]
    i1 = np.where(data_gnss["t"] >= t - .00001)[0][0]
    if i0 == i1:
        # Convert satellite positions to ENU coordinates
        N_sat = data_gnss["pr"][i0].shape[0]
        sat_pos_k = np.zeros((N_sat, 3))
        for j in range(data_gnss["sat_pos"][i0].shape[0]):
            sat_pos_ENU = utils.ecef2enu(data_gnss["sat_pos"][i0][j,:], p_ref_ECEF)
            sat_pos_k[j,:] = sat_pos_ENU
        R_k = np.diag(r_pr*np.ones(data_gnss["pr"][i0].shape[0]))
        pr_k = data_gnss["pr"][i0]
    else:
        sat_pos_k = None
        R_k = None
        pr_k = None

    # Update EKF using measurement and control
    u_k = traj["u"][:,k]
    dparams = {"dt":dt, "car_params":car_params}
    mparams = {"sat_pos":sat_pos_k}
    ekf_filter.update(u_k, pr_k, Q_EKF, R_k, dyn_func_params=dparams, meas_func_params=mparams)

    # Check for divergence
    if check_for_divergence(ekf_filter.mu[:2], traj["x"][:2,k]):
        EKF["t"] = EKF["t"][:k]
        EKF["bias"] = EKF["bias"][:k]
        EKF["x_ENU"] = EKF["x_ENU"][:k]
        EKF["y_ENU"] = EKF["y_ENU"][:k]
        EKF["z_ENU"] = EKF["z_ENU"][:k]
        EKF["psi"] = EKF["psi"][:k]
        EKF["vx"] = EKF["vx"][:k]
        EKF["vy"] = EKF["vy"][:k]
        break
data_utils.save_obj(EKF, car_path + '/filtering/ekf')

################################################################
# NLP using 2 norm
################################################################
T = 2 # finite time horizon for NLP
N = 5 # number of pseudospectral nodes
n = 9 # state is x = [px, py, psi, vx, vy, psid, b, bd, pz]
m = 2 # control is [throttle, steer]
dt_gnss = data_gnss["t"][1] - data_gnss["t"][0]
N_sat = 11
problem = nlp.fixedTimeOptimalEstimationNLP(N, T, n, m)
X = problem.addVariables(N+1, n, name='x')
U, W = problem.addDynamics(dynamics.vehicle_dynamics_and_gnss, X, None, None, {"car_params":car_params})
problem.addDynamicsCost(cost_functions.weighted_l2_norm, W, {"Q":np.linalg.inv(Q_NLP)})
problem.addVarBounds(X, 2, -np.pi, np.pi)
problem.addVarBounds(X, 3, 0, np.inf)
X0 = problem.addInitialCost(cost_functions.weighted_l2_norm, X[0], {"Q":np.linalg.inv(P_NLP)})

# Add costs for the pseudorange measurements at sampling frequency
N_gnss = int(np.floor(T/dt_gnss))
t_gnss = np.linspace(0, T, N_gnss + 1)
Y = []
R = []
sat_pos = []
for i in range(N_gnss + 1):
    t_i = np.array([[t_gnss[i]]])
    Y_i = []
    R_i = []
    sat_pos_i = []
    for j in range(N_sat):
        sat_pos_ij = problem.addParameter(1, 3)[0]
        R_ij = problem.addParameter(1, 1)[0]
        Y_ij = problem.addResidualCost(measurements.vehicle_pseudorange, X, t_i, None, 
                    R_ij, {"p":1, "sat_pos":sat_pos_ij})[0]
        Y_i.append(Y_ij)
        R_i.append(R_ij)
        sat_pos_i.append(sat_pos_ij)
    Y.append(Y_i)
    R.append(R_i)
    sat_pos.append(sat_pos_i)
problem.build()

NLP = {"t":[], "p_ref_ECEF":p_ref_ECEF, "bias":[],
          "x_ENU":[], "y_ENU":[], "z_ENU":[],
          "lat":[], "lon":[], "h":[],
          "psi":[], "vx":[], "vy":[], "t_solve":[]}

# Run NLP filter
DT = 1 # s, how often to recompute
tf = 89 # s, final time to stop computing
compute_times = np.linspace(0, tf, np.floor(tf/DT) + 1)
xhat0 = np.hstack((traj["x0"], np.array([data_gnss["b0"], data_gnss["alpha"], 0.0])))
for (step, t0) in enumerate(compute_times):
    traj_indices = utils.get_time_indices(traj["t"], t0, t0+T)
    gnss_indices = utils.get_time_indices(data_gnss["t"], t0, t0+T)
    traj_shifted_times = traj["t"][traj_indices] - t0
    gnss_shifted_times = data_gnss["t"][gnss_indices] - t0

    # Define control inputs
    u = traj["u"][:, traj_indices]
    problem.setControl(U, traj_shifted_times, u)

    # Set the initial condition
    problem.setParameter(X0, xhat0)

    # Specify the measurements
    for i in range(N_gnss + 1):
        i_gnss = gnss_indices[i]
        t_i = np.array([[t_gnss[i]]])
        N_sat_i = data_gnss["sat_pos"][i].shape[0]
        for j in range(N_sat):
            # Not every time step will have N_sat measurements, so set some costs to 0
            if j < N_sat_i:
                sat_pos_ENU = utils.ecef2enu(data_gnss["sat_pos"][i_gnss][j,:], p_ref_ECEF)
                R_ij = dt_gnss*np.linalg.inv(np.diag([r_pr])) # multiply by dt to integrate over the sampling interval
                problem.setParameter(R[i][j], R_ij)
                problem.setParameter(sat_pos[i][j], sat_pos_ENU)

                y_ij = np.array([[data_gnss["pr"][i_gnss][j]]])
                problem.setMeasurement(Y[i][j], t_i, y_ij)
            else:
                problem.setParameter(R[i][j], 0.0)
                problem.setParameter(sat_pos[i][j], np.zeros(3))
                problem.setMeasurement(Y[i][j], t_i, np.array([[0.0]]))

    # Solve problem
    print('Solving problem.')
    problem.solve(warmstart=True)

    t = np.linspace(0, T, 10)
    x_opt = problem.extractSolution('x', t) # full trajectory sampling
    xhat0 = problem.extractSolution('x', [DT]) # next initialization point
    # Store solution information
    for k in range(t.shape[0]):
        p_ENU = np.array([x_opt[k, 0], x_opt[k, 1], x_opt[k, 8]])
        p_ECEF = utils.enu2ecef(p_ENU, p_ref_ECEF)
        p_LLA = utils.ecef2lla(p_ECEF)
        NLP["x_ENU"].append(p_ENU[0])
        NLP["y_ENU"].append(p_ENU[1])
        NLP["z_ENU"].append(p_ENU[2])
        NLP["lat"].append(p_LLA[0])
        NLP["lon"].append(p_LLA[1])
        NLP["h"].append(p_LLA[2])
        NLP["bias"].append(x_opt[k,6])
        NLP["psi"].append(x_opt[k,2])
        NLP["vx"].append(x_opt[k,3])
        NLP["vy"].append(x_opt[k,4])
        NLP["t"].append(t0 + t[k])
    NLP["t_solve"].append(problem.solver["t_wall_total"])
data_utils.save_obj(NLP, car_path + '/filtering/nlp-l2')


################################################################
# NLP using huber loss
################################################################
problem = nlp.fixedTimeOptimalEstimationNLP(N, T, n, m)
X = problem.addVariables(N+1, n, name='x')
U, W = problem.addDynamics(dynamics.vehicle_dynamics_and_gnss, X, None, None, {"car_params":car_params})
problem.addDynamicsCost(cost_functions.pseudo_huber_loss, W, {"Q":np.linalg.inv(Q_NLP), "delta":5.0})
problem.addVarBounds(X, 2, -np.pi, np.pi)
problem.addVarBounds(X, 3, 0, np.inf)
X0 = problem.addInitialCost(cost_functions.weighted_l2_norm, X[0], {"Q":np.linalg.inv(P_NLP)})

# Add costs for the pseudorange measurements at sampling frequency
Y = []
R = []
sat_pos = []
for i in range(N_gnss + 1):
    t_i = np.array([[t_gnss[i]]])
    Y_i = []
    R_i = []
    sat_pos_i = []
    for j in range(N_sat):
        sat_pos_ij = problem.addParameter(1, 3)[0]
        R_ij = problem.addParameter(1, 1)[0]
        Y_ij = problem.addResidualCost(measurements.vehicle_pseudorange, X, t_i, None, 
                    R_ij, {"p":1, "sat_pos":sat_pos_ij})[0]
        Y_i.append(Y_ij)
        R_i.append(R_ij)
        sat_pos_i.append(sat_pos_ij)
    Y.append(Y_i)
    R.append(R_i)
    sat_pos.append(sat_pos_i)
problem.build()

NLP = {"t":[], "p_ref_ECEF":p_ref_ECEF, "bias":[],
          "x_ENU":[], "y_ENU":[], "z_ENU":[],
          "lat":[], "lon":[], "h":[],
          "psi":[], "vx":[], "vy":[], "t_solve":[]}

# Run NLP filter
xhat0 = np.hstack((traj["x0"], np.array([data_gnss["b0"], data_gnss["alpha"], 0.0])))
for (step, t0) in enumerate(compute_times):
    traj_indices = utils.get_time_indices(traj["t"], t0, t0+T)
    gnss_indices = utils.get_time_indices(data_gnss["t"], t0, t0+T)
    traj_shifted_times = traj["t"][traj_indices] - t0
    gnss_shifted_times = data_gnss["t"][gnss_indices] - t0

    # Define control inputs
    u = traj["u"][:, traj_indices]
    problem.setControl(U, traj_shifted_times, u)

    # Set the initial condition
    problem.setParameter(X0, xhat0)

    # Specify the measurements
    M_gnss = (N_gnss + 1)*N_sat # a value to normalize the cost of the measurements
    for i in range(N_gnss + 1):
        i_gnss = gnss_indices[i]
        t_i = np.array([[t_gnss[i]]])
        N_sat_i = data_gnss["sat_pos"][i].shape[0]
        for j in range(N_sat):
            # Not every time step will have N_sat measurements, so set some costs to 0
            if j < N_sat_i:
                sat_pos_ENU = utils.ecef2enu(data_gnss["sat_pos"][i_gnss][j,:], p_ref_ECEF)
                R_ij = dt_gnss*np.linalg.inv(np.diag([r_pr]))
                problem.setParameter(R[i][j], R_ij)
                problem.setParameter(sat_pos[i][j], sat_pos_ENU)

                y_ij = np.array([[data_gnss["pr"][i_gnss][j]]])
                problem.setMeasurement(Y[i][j], t_i, y_ij)
            else:
                problem.setParameter(R[i][j], 0.0)
                problem.setParameter(sat_pos[i][j], np.zeros(3))
                problem.setMeasurement(Y[i][j], t_i, np.array([[0.0]]))

    # Solve problem
    print('Solving problem.')
    problem.solve(warmstart=True)

    t = np.linspace(0, T, 10)
    x_opt = problem.extractSolution('x', t) # full trajectory sampling
    xhat0 = problem.extractSolution('x', [DT]) # next initialization point
    for k in range(t.shape[0]):
        p_ENU = np.array([x_opt[k, 0], x_opt[k, 1], x_opt[k, 8]])
        p_ECEF = utils.enu2ecef(p_ENU, p_ref_ECEF)
        p_LLA = utils.ecef2lla(p_ECEF)
        NLP["x_ENU"].append(p_ENU[0])
        NLP["y_ENU"].append(p_ENU[1])
        NLP["z_ENU"].append(p_ENU[2])
        NLP["lat"].append(p_LLA[0])
        NLP["lon"].append(p_LLA[1])
        NLP["h"].append(p_LLA[2])
        NLP["bias"].append(x_opt[k,6])
        NLP["psi"].append(x_opt[k,2])
        NLP["vx"].append(x_opt[k,3])
        NLP["vy"].append(x_opt[k,4])
        NLP["t"].append(t0 + t[k])
    NLP["t_solve"].append(problem.solver["t_wall_total"])
data_utils.save_obj(NLP, car_path + '/filtering/nlp-huber')