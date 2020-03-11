import numpy as np
import os
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
import pdb

dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(dir_path, 'data/multi-receiver')

# Reference location on Earth (Hoover Tower)
lat0 = 37.4276
lon0 = -122.1670
h0 = 0
p_ref_ECEF = utils.lla2ecef(np.array([lat0, lon0, h0]))

# Data Loading, data_A (receiver 1) is stationary and data_B (receiver 2) moves
data_A = data_utils.load_gnss_logs(data_path + '/rec1_gnss_log_2020_02_27_13_07_10')
data_B = data_utils.load_gnss_logs(data_path + '/rec2_gnss_log_2020_02_27_13_08_17')

# Synchronize time from both to start at 0
t0 = np.min(np.hstack((data_A["t"], data_B["t"])))
data_A["t"] -= t0
data_B["t"] -= t0

# Compute iterative least squares solutions
LS_A = ls.runLeastSquares(data_A["t"], data_A["sat_pos"], data_A["pr"], data_A["sat_vel"], data_A["pr_rate"], p_ref_ECEF)
LS_B = ls.runLeastSquares(data_B["t"], data_B["sat_pos"], data_B["pr"], data_B["sat_vel"], data_B["pr_rate"], p_ref_ECEF)

# Compute mean of receiver 1 data
LSmean_A = {"p_ref_ECEF":p_ref_ECEF}
LSmean_A["x_ENU"] = np.mean(LS_A["x_ENU"])
LSmean_A["y_ENU"] = np.mean(LS_A["y_ENU"])
LSmean_A["z_ENU"] = np.mean(LS_A["z_ENU"])

# Batch least squares assuming no motion for receiver 2, from start (t0 = 58) to t = 90 seconds
idx = utils.get_time_indices(data_B["t"], data_B["t"][0], 90)
LSbatch_B = ls.runBatchLeastSquares(data_B["t"][idx], [data_B["sat_pos"][i] for i in idx], [data_B["pr"][i] for i in idx], p_ref_ECEF)



################################################################
# Cost matrices for EKF and NLP
################################################################
Q = np.diag([0.01, 0.01, 0.01, 0.01, 1., 1., 0.01, 0.01]) # covariance for dynamics
r_pr = 1000 # covariance for pseudorange measurement
r_prr = 100
r_range = 0.00001
indices = utils.get_time_indices(data_B["t"], LSbatch_B["t"][-1], data_B["t"][-50])

# ################################################################
# # EKF
# ################################################################
# N_steps = len(indices)
# EKF = {"t":data_B["t"][indices], "p_ref_ECEF":p_ref_ECEF, "bias":np.zeros(N_steps),
#       "x_ENU":np.zeros(N_steps), "y_ENU":np.zeros(N_steps), "z_ENU":np.zeros(N_steps),
#       "lat":np.zeros(N_steps), "lon":np.zeros(N_steps), "h":np.zeros(N_steps)}

# # Create EKF object
# xhat0 = np.array([LSbatch_B["x_ENU"], LSbatch_B["y_ENU"], LSbatch_B["z_ENU"], LSbatch_B["b0"], LSbatch_B["alpha"]]) # initialize estimate using Least squares solution
# P0 = np.diag([1, 1, 1, 1, 1]) # initialize covariance
# ekf_filter = ekf.EKF(gnss.gnss_pos_and_bias, gnss.multi_pseudorange, xhat0, P0)

# # Run EKF
# for (k, i) in enumerate(indices):
#     EKF["x_ENU"][k] = ekf_filter.mu[0]
#     EKF["y_ENU"][k] = ekf_filter.mu[1]
#     EKF["z_ENU"][k] = ekf_filter.mu[2]
#     EKF["bias"][k] = ekf_filter.mu[3]

#     # Convert satellite positions to ENU coordinates
#     sat_pos_i = np.array([]).reshape(0,3)
#     for j in range(data_B["sat_pos"][i].shape[0]):
#         sat_pos_ENU = utils.ecef2enu(data_B["sat_pos"][i][j,:], p_ref_ECEF)
#         sat_pos_i = np.vstack((sat_pos_i, sat_pos_ENU.reshape((1,3))))

#     # Update EKF using measurement and control
#     u_i = np.array([LS_B["xd_ENU"][i+1], LS_B["yd_ENU"][i+1], LS_B["zd_ENU"][i+1]])
#     R = np.diag(r_pr*np.ones(data_B["pr"][i].shape[0]))
#     dt = LS_B["t"][i+1] - LS_B["t"][i]
#     ekf_filter.update(u_i, data_B["pr"][i], Q, R, dyn_func_params={"dt":dt}, meas_func_params={"sat_pos":sat_pos_i})


################################################################
# NLP
################################################################
N = 15 # number of pseudospectral nodes
n = 8 # state is x = [xB, yB, zB, bB, xdB, ydB, zdB, alphaB]
m = 0 # no control input
t0 = data_B["t"][indices][0]
tf = data_B["t"][indices[-1]]
shifted_times = data_B["t"][indices] - t0

problem = nlp.fixedTimeOptimalEstimationNLP(N, tf - t0, n, m)

# Define variables
XB = problem.addVariables(N+1, n, name='x')
XA = problem.addVariables(1, 3, name='xa')[0]

# Define system dynamics
_, W = problem.addDynamics(dynamics.multi_receiver, XB)
Q_NLP = np.linalg.inv(Q)
for i in [4, 5]:
    Q_NLP[i,i] = 0 # no cost for W referencing velocity dynamics being large
problem.addDynamicsCost(cost_functions.weighted_l2_norm, W, {"Q":np.linalg.inv(Q)})

# Add pseudorange (rate) measurement costs
for (k, i) in enumerate(indices):
    t_i = np.array([[shifted_times[k]]])
    for (j, pr) in enumerate(data_B["pr"][i]):
        sat_pos_ENU = utils.ecef2enu(data_B["sat_pos"][i][j,:], p_ref_ECEF)
        sat_vel_ENU = utils.ecef2enu(data_B["sat_vel"][i][j,:], p_ref_ECEF, rotation_only=True)
        params = {"sat_pos": sat_pos_ENU, "sat_vel": sat_vel_ENU}
        R_ij = np.diag([r_pr])
        y_ij = np.array([[pr]])
        problem.addResidualCost(measurements.pseudorange, XB, t_i, y_ij, np.linalg.inv(R_ij), params)

        Rr_ij = np.diag([r_prr])
        yr_ij = np.array([[data_B["pr_rate"][i][j]]])
        problem.addResidualCost(measurements.pseudorange_rate, XB, t_i, yr_ij, np.linalg.inv(Rr_ij), params)
            
    Rrange_ij = np.diag([r_range])
    yrange_ij = np.array([[2.4384]])
    problem.addResidualCost(measurements.multi_receiver_range, XB, t_i, yrange_ij, np.linalg.inv(Rrange_ij), {"y":XA})


# Solve problem
print('Building problem.')
problem.build()

print('Solving problem.')
problem.solve()
# problem.solve(warmstart=True)

N_steps = 100
t = np.linspace(0, tf-t0, N_steps)

NLP = {"t":t+t0, "p_ref_ECEF":p_ref_ECEF, "bias":np.zeros(N_steps),
      "x_ENU":np.zeros(N_steps), "y_ENU":np.zeros(N_steps), "z_ENU":np.zeros(N_steps),
      "xd_ENU":np.zeros(N_steps), "yd_ENU":np.zeros(N_steps), "zd_ENU":np.zeros(N_steps),
      "lat":np.zeros(N_steps), "lon":np.zeros(N_steps), "h":np.zeros(N_steps)}

x_opt = problem.extractSolution('x', t)
for k in range(N_steps):
    p_ENU = np.array([x_opt[k, 0], x_opt[k, 1], x_opt[k, 2]])
    pd_ENU = np.array([x_opt[k, 4], x_opt[k, 5], x_opt[k, 6]])
    p_ECEF = utils.enu2ecef(p_ENU, p_ref_ECEF)
    p_LLA = utils.ecef2lla(p_ECEF)
    NLP["x_ENU"][k] = p_ENU[0]
    NLP["y_ENU"][k] = p_ENU[1]
    NLP["z_ENU"][k] = p_ENU[2]
    NLP["bias"][k] = x_opt[k,3]
    NLP["xd_ENU"][k] = pd_ENU[0]
    NLP["yd_ENU"][k] = pd_ENU[1]
    NLP["zd_ENU"][k] = pd_ENU[2]
    NLP["lat"][k] = p_LLA[0]
    NLP["lon"][k] = p_LLA[0]
    NLP["h"][k] = p_LLA[0]
    


# Plotting
plt.figure(1)
# plt.scatter(LSmean_A["x_ENU"], LSmean_A["y_ENU"], c='r', marker='x', label='LS_A')
# plt.scatter(LS_B["x_ENU"], LS_B["y_ENU"], c='r', marker='o', label='LS_B')
plt.scatter(LSbatch_B["x_ENU"], LSbatch_B["y_ENU"], c='k', marker='s', label='Batch LS_B')
# plt.scatter(EKF["x_ENU"], EKF["y_ENU"], c='g', marker='d', label='EKF')
plt.scatter(NLP["x_ENU"], NLP["y_ENU"], c='b', marker='o', label='NLP')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.legend()

plt.figure(2)
# plt.plot(LS_B["t"], LS_B["x_ENU"], c='r', label='x (LS)')
plt.plot(LSbatch_B["t"], LSbatch_B["x_ENU"]*np.ones(LSbatch_B["t"].shape), c='k', label='x (Batch LS)')
# plt.plot(EKF["t"], EKF["x_ENU"], c='g', label='x (EKF)')
plt.plot(NLP["t"], NLP["x_ENU"], c='b', label='x (NLP)')
plt.xlabel('t (s)')
plt.ylabel('x (m)')
plt.legend()

plt.figure(3)
# plt.plot(LS_B["t"], LS_B["y_ENU"], c='r', label='y (LS)')
plt.plot(LSbatch_B["t"], LSbatch_B["y_ENU"]*np.ones(LSbatch_B["t"].shape), c='k', label='y (Batch LS)')
# plt.plot(EKF["t"], EKF["y_ENU"], c='g', label='y (EKF)')
plt.plot(NLP["t"], NLP["y_ENU"], c='b', label='y (NLP)')
plt.xlabel('t (s)')
plt.ylabel('y (m)')
plt.legend()

plt.figure(4)

plt.plot(LS_B["t"], LS_B["xd_ENU"], c='r', label='x (LS_B)')
plt.plot(LS_B["t"], LS_B["yd_ENU"], c='b', label='y (LS_B)')
plt.plot(NLP["t"], NLP["xd_ENU"], c='k', label='xd (NLP)')
plt.plot(NLP["t"], NLP["yd_ENU"], c='g', label='xy (NLP)')
plt.xlabel('t (s)')
plt.ylabel('v (m/s)')
plt.legend()


plt.show()

pdb.set_trace()



