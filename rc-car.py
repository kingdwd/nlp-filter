import numpy as np
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

# Reference location on Earth (Hoover Tower)
lat0 = 37.4276
lon0 = -122.1670
h0 = 0
p_ref_ECEF = utils.lla2ecef(np.array([lat0, lon0, h0]))

# Load data
data_gnss = data_utils.load_gnss_logs('./data/rc-car/gnss/gnss_log_2020_02_27_10_02_20')
data_px4 = data_utils.load_px4_logs('./data/rc-car/px4/log_164_2020-2-27-10-03-56')

# Set control throttle to zero below threshold for modeling purposes
for i in range(data_px4["u"].shape[1]):
    if data_px4["u"][0,i] < 0.1:
        data_px4["u"][0,i] = 0.0

# Synchronize time from both to start at 0
for (k, t) in enumerate(data_px4["t"]):
    if np.abs(data_px4["u"][1,k]) < 0.01:
        print('Starting time logged as {}'.format(t))
        data_px4["t"] = data_px4["t"][k:]
        data_px4["u"] = data_px4["u"][:,k:]
        break
data_gnss["t"] -= data_gnss["t"][0]
data_px4["t"] -= data_px4["t"][0]

# Compute iterative least squares solutions
LS = ls.runLeastSquares(data_gnss["t"], data_gnss["sat_pos"], data_gnss["pr"], data_gnss["sat_vel"], data_gnss["pr_rate"], p_ref_ECEF)


################################################################
# Cost matrices for EKF and NLP
################################################################
Q = np.diag([1, 1, 0.001, .01, .01, 1]) # covariance for dynamics
r_pr = 10 # covariance for pseudorange measurement

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
t0 = 0.0
tf = 40.0
gnss_indices = utils.get_time_indices(data_gnss["t"], t0, tf)
px4_indices = utils.get_time_indices(data_px4["t"], t0, tf)
N = 20 # number of pseudospectral nodes
n = 6 # state is x = [x, y, z, b, bd, th]
m = 2 # control is throttle, steer

gnss_shifted_times = data_gnss["t"][gnss_indices] - t0
px4_shifted_times = data_px4["t"][px4_indices] - t0

problem = nlp.fixedTimeOptimalEstimationNLP(N, tf - t0, n, m)

# Define variables
X = problem.addVariables(N+1, n, name='x')

# Define system dynamics
u = data_px4["u"][:, px4_indices]
_, W = problem.addDynamics(dynamics.kinematic_bycicle_and_bias, X, px4_shifted_times, u)
problem.addDynamicsCost(cost_functions.weighted_l2_norm, W, {"Q":np.linalg.inv(Q)})

# Define cost function, adding measurements individually
for (k, i) in enumerate(gnss_indices):
    t_i = np.array([[gnss_shifted_times[k]]])
    for (j, pr) in enumerate(data_gnss["pr"][i]):
        sat_pos_ENU = utils.ecef2enu(data_gnss["sat_pos"][i][j,:], p_ref_ECEF)
        params = {"sat_pos": sat_pos_ENU}
        R = np.diag([r_pr])
        y = np.array([[pr]])
        problem.addResidualCost(measurements.pseudorange, X, t_i, y, np.linalg.inv(R), params)

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
      "lat":np.zeros(N_steps), "lon":np.zeros(N_steps), "h":np.zeros(N_steps)}

x_opt = problem.extractSolution('x', t)
for k in range(N_steps):
    p_ENU = np.array([x_opt[k, 0], x_opt[k, 1], x_opt[k, 2]])
    p_ECEF = utils.enu2ecef(p_ENU, p_ref_ECEF)
    p_LLA = utils.ecef2lla(p_ECEF)
    NLP["x_ENU"][k] = p_ENU[0]
    NLP["y_ENU"][k] = p_ENU[1]
    NLP["z_ENU"][k] = p_ENU[2]
    NLP["lat"][k] = p_LLA[0]
    NLP["lon"][k] = p_LLA[0]
    NLP["h"][k] = p_LLA[0]
    NLP["bias"][k] = x_opt[k,3]


# Plotting
plt.figure(1)
plt.scatter(LS["x_ENU"], LS["y_ENU"], c='r', marker='o', label='LS')
# plt.scatter(LS_batch_A["x_ENU"], LS_batch_A["y_ENU"], c='k', marker='d', label='Batch LS_A')
# plt.scatter(LSbatch_B["x_ENU"], LSbatch_B["y_ENU"], c='k', marker='s', label='Batch LS_B')
# plt.scatter(EKF["x_ENU"], EKF["y_ENU"], c='g', marker='d', label='EKF')
plt.scatter(NLP["x_ENU"], NLP["y_ENU"], c='b', marker='o', label='NLP')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.legend()

plt.figure(2)
plt.plot(LS["t"], LS["x_ENU"], c='r', label='x (LS)')
# plt.plot(EKF["t"], EKF["x_ENU"], c='g', label='x (EKF)')
plt.plot(NLP["t"], NLP["x_ENU"], c='b', label='x (NLP)')
plt.xlabel('t (s)')
plt.ylabel('x (m)')
plt.legend()

plt.figure(3)
plt.plot(LS["t"], LS["y_ENU"], c='r', label='y (LS)')
# plt.plot(EKF["t"], EKF["y_ENU"], c='g', label='y (EKF)')
plt.plot(NLP["t"], NLP["y_ENU"], c='b', label='y (NLP)')
plt.xlabel('t (s)')
plt.ylabel('y (m)')
plt.legend()

plt.figure(4)
# plt.plot(LS_A["t"], LS_A["xd_ENU"], c='r', label='x (LS_A)')
# plt.plot(LS_A["t"], LS_A["yd_ENU"], c='b', label='y (LS_A)')
plt.plot(LS["t"], LS["xd_ENU"], c='r', label='x (LS)')
plt.plot(LS["t"], LS["yd_ENU"], c='b', label='y (LS)')
plt.xlabel('t (s)')
plt.ylabel('v (m/s)')
plt.legend()


plt.show()



