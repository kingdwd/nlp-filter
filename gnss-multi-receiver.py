import numpy as np
from scipy.interpolate import interp1d
import os
import matplotlib.pyplot as plt
import nlp.nlp as nlp
import nlp.dynamics as dynamics
import nlp.cost_functions as cost_functions
import nlp.constraints as constraints
import nlp.simulate as simulate
import nlp.measurements as measurements
import utils.gnss as gnss
import utils.leastsquares as ls
import utils.utils as utils
import utils.ekf as ekf
import utils.data as data_utils

dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(dir_path, 'data/gnss-multi-receiver')
dataA = data_utils.load_gnss_logs(data_path + '/rec1/rec1_gnss_log_50y_moving_')
dataB = data_utils.load_gnss_logs(data_path + '/rec2/rec2_gnss_log_50y_moving_')

# Reference location on Earth (Hoover Tower)
lat0 = 37.4276
lon0 = -122.1670
h0 = 0
p_ref_ECEF = utils.lla2ecef(np.array([lat0, lon0, h0]))

# Synchronize time from both to start at 0
t0 = np.min(np.hstack((dataA["t"], dataB["t"])))
dataA["t"] -= t0
dataB["t"] -= t0

# Compute iterative least squares solutions
LS_A = ls.runLeastSquares(dataA["t"], dataA["sat_pos"], dataA["pr"], dataA["sat_vel"], dataA["pr_rate"], p_ref_ECEF)
LS_B = ls.runLeastSquares(dataB["t"], dataB["sat_pos"], dataB["pr"], dataB["sat_vel"], dataB["pr_rate"], p_ref_ECEF)
data_utils.save_obj(LS_A, data_path + '/filtering/leastsquaresA')
data_utils.save_obj(LS_B, data_path + '/filtering/leastsquaresB')

################################################################
# NLP
################################################################
# Cost
Q = np.diag([.01, .01, .01, 0.01, 0.01, .01, .01, .01, 0.01, 0.01])
P = 0.01*np.diag([1, 1, 1, 0.1, 0.1, 1, 1, 1, 0.1, 0.1])
r_pr_A = 10
r_pr_B = 1
r_range = 0.01
r_heading = 0.1

# Time horizon and problem definition
T = 5 # finite time horizon for NLP, seconds
N = 10
n = 10 # state is x = [xA, yA, zA, bA, alphaA, xB, yB, zB, bB, alphaB]
m = 6 # control is u = [vxA, vyA, vzA, vxB, vyB, vzB]

problem = nlp.fixedTimeOptimalEstimationNLP(N, T, n, m)

# Define variables
X = problem.addVariables(N+1, n, name='x')

# Define system dynamics
U, W = problem.addDynamics(dynamics.gnss_two_receiver, X, None, None)
problem.addDynamicsCost(cost_functions.weighted_l2_norm, W, {"Q":np.linalg.inv(Q)})
X0 = problem.addInitialCost(cost_functions.weighted_l2_norm, X[0], {"Q":np.linalg.inv(P)})

# Add range measurement
distance = 0.5*91.44 # meters (50 yards)
dt_range = 0.1 # seconds
N_range = int(np.floor(T/dt_range))
t_range = np.linspace(0, T, N_range + 1)
y_range = distance*np.ones((1, t_range.shape[0]))
problem.addResidualCost(measurements.multi_receiver_range_3d, X, t_range, y_range, 
                                dt_range*np.array([1./r_range]), {"idxA":[0, 1, 2], "idxB":[5, 6, 7]})

# Add zA = zB constraint
for i in range(N + 1):
    args = [X[i][2], X[i][7]]
    problem.addEqConstraint(constraints.equality_constaint, args)

# Add heading measurement
heading = -44 # degrees, counterclockwise from East
dt_heading = 0.1 # seconds
N_heading = int(np.floor(T/dt_heading))
t_heading = np.linspace(0, T, N_heading + 1)
y_heading = np.deg2rad(heading)*np.ones((1, t_heading.shape[0]))
problem.addResidualCost(measurements.multi_receiver_heading_2d, X, t_heading, y_heading, 
                                dt_heading*np.array([1./r_heading]), {"idxA":[0, 1], "idxB":[5, 6]})

# Add costs for the pseudorange measurements at sampling frequency
dt_gnss = 1
N_sat = 10
N_gnss = int(np.floor(T/dt_gnss))
t_gnss = np.linspace(0, T, N_gnss + 1)
R_A = []
Y_A = []
sat_pos_A = []
R_B = []
Y_B = []
sat_pos_B = []
for i in range(N_gnss + 1):
    t_i = np.array([[t_gnss[i]]])
    Y_i_A = []
    R_i_A = []
    sat_pos_i_A = []
    Y_i_B = []
    R_i_B = []
    sat_pos_i_B = []
    for j in range(N_sat):
        # Receiver A adding psuedorange measurements
        sat_pos_ij_A = problem.addParameter(1, 3)[0]
        R_ij_A = problem.addParameter(1, 1)[0]
        Y_ij_A = problem.addResidualCost(measurements.pseudorange, X, t_i, None, 
                    R_ij_A, {"p":1, "sat_pos":sat_pos_ij_A, "idx":[0,1,2,3]})[0]
        Y_i_A.append(Y_ij_A)
        R_i_A.append(R_ij_A)
        sat_pos_i_A.append(sat_pos_ij_A)

        # Receiver B adding psuedorange measurements
        sat_pos_ij_B = problem.addParameter(1, 3)[0]
        R_ij_B = problem.addParameter(1, 1)[0]
        Y_ij_B = problem.addResidualCost(measurements.pseudorange, X, t_i, None, 
                    R_ij_B, {"p":1, "sat_pos":sat_pos_ij_B, "idx":[5,6,7,8]})[0]
        Y_i_B.append(Y_ij_B)
        R_i_B.append(R_ij_B)
        sat_pos_i_B.append(sat_pos_ij_B)
    Y_A.append(Y_i_A)
    R_A.append(R_i_A)
    sat_pos_A.append(sat_pos_i_A)
    Y_B.append(Y_i_B)
    R_B.append(R_i_B)
    sat_pos_B.append(sat_pos_i_B)
problem.build()

NLP = {"t":[], "p_ref_ECEF":p_ref_ECEF, "biasA":[], "biasB":[],
          "xA_ENU":[], "yA_ENU":[], "zA_ENU":[],
          "latA":[], "lonA":[], "hA":[],
          "xB_ENU":[], "yB_ENU":[], "zB_ENU":[],
          "latB":[], "lonB":[], "hB":[], "t_solve":[]}

# Run NLP filter
xhat0 = np.array([LS_A["x_ENU"][0], LS_A["y_ENU"][0], LS_A["z_ENU"][0], LS_A["bias"][0], 0.0,
                  LS_B["x_ENU"][0], LS_B["y_ENU"][0], LS_B["z_ENU"][0], LS_B["bias"][0], 0.0,])
DT = 1 # s, how often to recompute
tf = 95 # s, final time to stop computing
compute_times = np.linspace(0, tf, np.floor(tf/DT) + 1)
t_offset = dataB["t"][0] - dataA["t"][0]
if np.abs(t_offset) > .001:
    print('times are not synchronized')
for (step, t0) in enumerate(compute_times):
    gnssA_indices = utils.get_time_indices(dataA["t"], t0, t0+T)
    gnssB_indices = utils.get_time_indices(dataB["t"], t0 + t_offset, t0 + t_offset + T)
    gnssA_shifted_times = dataA["t"][gnssA_indices] - t0
    gnssB_shifted_times = dataB["t"][gnssB_indices] - t0 - t_offset

    # Define control inputs
    uA = np.vstack((LS_A["xd_ENU"][gnssA_indices].reshape(1,-1),
                    LS_A["yd_ENU"][gnssA_indices].reshape(1,-1),
                    LS_A["zd_ENU"][gnssA_indices].reshape(1,-1)))
    uB = np.vstack((LS_B["xd_ENU"][gnssB_indices].reshape(1,-1),
                    LS_B["yd_ENU"][gnssB_indices].reshape(1,-1),
                    LS_B["zd_ENU"][gnssB_indices].reshape(1,-1)))
    f = interp1d(gnssB_shifted_times, uB, fill_value="extrapolate")
    uB = f(gnssA_shifted_times)
    u = np.vstack((uA, uB))
    problem.setControl(U, gnssA_shifted_times, u)

    # Set initial condition
    problem.setParameter(X0, xhat0)

    # Specify the measurements for receiver A
    for i in range(N_gnss + 1):
        i_gnssA = gnssA_indices[i]
        t_i = np.array([[t_gnss[i]]])
        N_sat_i = dataA["sat_pos"][i_gnssA].shape[0]
        for j in range(N_sat):
            # Not every time step will have N_sat measurements, so set some costs to 0
            if j < N_sat_i:
                sat_pos_ENU = utils.ecef2enu(dataA["sat_pos"][i_gnssA][j,:], p_ref_ECEF)
                R_ij = dt_gnss*np.linalg.inv(np.diag([r_pr_A])) # multiply by dt to integrate over the sampling interval
                problem.setParameter(R_A[i][j], R_ij)
                problem.setParameter(sat_pos_A[i][j], sat_pos_ENU)

                y_ij = np.array([[dataA["pr"][i_gnssA][j]]])
                problem.setMeasurement(Y_A[i][j], t_i, y_ij)
            else:
                problem.setParameter(R_A[i][j], 0.0)
                problem.setParameter(sat_pos_A[i][j], np.zeros(3))
                problem.setMeasurement(Y_A[i][j], t_i, np.array([[0.0]]))

    # Specify the measurements for receiver B
    for i in range(N_gnss + 1):
        i_gnssB = gnssB_indices[i]
        t_i = np.array([[t_gnss[i]]])
        N_sat_i = dataB["sat_pos"][i_gnssB].shape[0]
        for j in range(N_sat):
            # Not every time step will have N_sat measurements, so set some costs to 0
            if j < N_sat_i:
                sat_pos_ENU = utils.ecef2enu(dataB["sat_pos"][i_gnssB][j,:], p_ref_ECEF)
                R_ij = dt_gnss*np.linalg.inv(np.diag([r_pr_B])) # multiply by dt to integrate over the sampling interval
                problem.setParameter(R_B[i][j], R_ij)
                problem.setParameter(sat_pos_B[i][j], sat_pos_ENU)

                y_ij = np.array([[dataB["pr"][i_gnssB][j]]])
                problem.setMeasurement(Y_B[i][j], t_i, y_ij)
            else:
                problem.setParameter(R_B[i][j], 0.0)
                problem.setParameter(sat_pos_B[i][j], np.zeros(3))
                problem.setMeasurement(Y_B[i][j], t_i, np.array([[0.0]]))

    # Solve problem
    print('Solving problem.')
    problem.solve(warmstart=True)

    t = np.linspace(0, T, 50)
    x_opt = problem.extractSolution('x', t) # full trajectory sampling
    xhat0 = problem.extractSolution('x', [DT]) # next initialization point

    # Store solution information
    k = t.shape[0] - 1
    p_ENU_A = np.array([x_opt[k, 0], x_opt[k, 1], x_opt[k, 2]])
    p_ENU_B = np.array([x_opt[k, 5], x_opt[k, 6], x_opt[k, 7]])
    p_ECEF_A = utils.enu2ecef(p_ENU_A, p_ref_ECEF)
    p_ECEF_B = utils.enu2ecef(p_ENU_B, p_ref_ECEF)
    p_LLA_A = utils.ecef2lla(p_ECEF_A)
    p_LLA_B = utils.ecef2lla(p_ECEF_B)
    NLP["xA_ENU"].append(p_ENU_A[0])
    NLP["yA_ENU"].append(p_ENU_A[1])
    NLP["zA_ENU"].append(p_ENU_A[2])
    NLP["latA"].append(p_LLA_A[0])
    NLP["lonA"].append(p_LLA_A[1])
    NLP["hA"].append(p_LLA_A[2])
    NLP["biasA"].append(x_opt[k,3])
    NLP["xB_ENU"].append(p_ENU_B[0])
    NLP["yB_ENU"].append(p_ENU_B[1])
    NLP["zB_ENU"].append(p_ENU_B[2])
    NLP["latB"].append(p_LLA_B[0])
    NLP["lonB"].append(p_LLA_B[1])
    NLP["hB"].append(p_LLA_B[2])
    NLP["biasB"].append(x_opt[k,8])
    NLP["t"].append(t0 + t[k])
    NLP["t_solve"].append(problem.solver["t_wall_total"])

data_utils.save_obj(NLP, data_path + '/filtering/nlp')


