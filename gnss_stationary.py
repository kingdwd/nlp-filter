import numpy as np
import os
import matplotlib.pyplot as plt
import nlp.nlp as nlp
import nlp.dynamics as dynamics
import nlp.cost_functions as cost_functions
import nlp.constraints as constraints
import nlp.measurements as measurements
import utils.gnss as gnss
import utils.leastsquares as ls
import utils.utils as utils
import utils.ekf as ekf
import utils.data as data_utils

dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(dir_path, 'data/gnss_stationary')

Q = np.diag([0.0001, 0.0001, 0.0001, 0.1, 0.001]) # covariance for dynamics
r_pr = 100 # covariance for pseudorange measurement
r_bias = 0.001

# Reference location on Earth (Hoover Tower)
lat0 = 37.4276
lon0 = -122.1670
h0 = 0
p_ref_ECEF = utils.lla2ecef(np.array([lat0, lon0, h0]))

# Assuming data sampled a 1 Hz
dt = 1
T = 50
t = np.linspace(0, T, T + 1)
u = np.zeros((3, T + 1))
data = data_utils.load_gnss_logs(data_path + '/gnss_log_2020_02_05_09_14_15')

# Compute iterative least squares solutions
LS = {"t":t, "p_ref_ECEF":p_ref_ECEF, "bias":np.zeros(T+1),
      "x_ENU":np.zeros(T+1), "y_ENU":np.zeros(T+1), "z_ENU":np.zeros(T+1),
      "lat":np.zeros(T+1), "lon":np.zeros(T+1), "h":np.zeros(T+1)}
for k in range(T+1):
    # Solve least squares
    p_ECEF, b = ls.iterativeLeastSquares(data["sat_pos"][k], data["pr"][k])
    p_ENU = utils.ecef2enu(p_ECEF, p_ref_ECEF)
    p_LLA = utils.ecef2lla(p_ECEF)
    LS["x_ENU"][k] = p_ENU[0]
    LS["y_ENU"][k] = p_ENU[1]
    LS["z_ENU"][k] = p_ENU[2]
    LS["lat"][k] = p_LLA[0]
    LS["lon"][k] = p_LLA[1]
    LS["h"][k] = p_LLA[2]
    LS["bias"][k] = b


# Compute iterative least squares using all measurements in a batch
sat_pos_batch = np.array([]).reshape(0,3)
pr_batch = np.array([])
t_batch = np.array([])
for k in range(T+1):
    sat_pos_batch = np.vstack((sat_pos_batch, data["sat_pos"][k]))
    pr_batch = np.hstack((pr_batch, data["pr"][k]))
    t_batch = np.hstack((t_batch, [t[k]]*data["pr"][k].shape[0]))

# Solve batch least squares 
p_ECEF, b0, alpha = ls.iterativeLeastSquares_multiTimeStep(t_batch, sat_pos_batch, pr_batch)
p_ENU = utils.ecef2enu(p_ECEF, p_ref_ECEF)
p_LLA = utils.ecef2lla(p_ECEF)
LS_batch = {"t":t, "p_ref_ECEF":p_ref_ECEF,
      "x_ENU":p_ENU[0], "y_ENU":p_ENU[1], "z_ENU":p_ENU[2],
      "lat":p_LLA[0], "lon":p_LLA[1], "h":p_LLA[2]}


# Compute using EKF
EKF = {"t":t, "p_ref_ECEF":p_ref_ECEF, "bias":np.zeros(T+1),
      "x_ENU":np.zeros(T+1), "y_ENU":np.zeros(T+1), "z_ENU":np.zeros(T+1),
      "lat":np.zeros(T+1), "lon":np.zeros(T+1), "h":np.zeros(T+1)}

# Create EKF object
bias_rate_guess = (LS["bias"][-1] - LS["bias"][0])/T

xhat0 = np.array([LS["x_ENU"][0], LS["y_ENU"][0], LS["z_ENU"][0], LS["bias"][0], bias_rate_guess]) # initialize estimate using Least squares solution
P0 = np.diag([1, 1, 1, 1, 1]) # initialize covariance
ekf_filter = ekf.EKF(gnss.gnss_pos_and_bias, gnss.multi_pseudorange, xhat0, P0)

# Run EKF
for k in range(T+1):
    EKF["x_ENU"][k] = ekf_filter.mu[0]
    EKF["y_ENU"][k] = ekf_filter.mu[1]
    EKF["z_ENU"][k] = ekf_filter.mu[2]
    EKF["bias"][k] = ekf_filter.mu[3]

    # Convert to ENU coordinates
    sat_pos_k = np.array([]).reshape(0,3)
    for i in range(data["sat_pos"][k].shape[0]):
        sat_pos_ENU = utils.ecef2enu(data["sat_pos"][k][i,:], p_ref_ECEF)
        sat_pos_k = np.vstack((sat_pos_k, sat_pos_ENU.reshape((1,3))))

    # Update EKF using measurement and control from next time step
    R = np.diag(r_pr*np.ones(data["pr"][k].shape[0]))
    ekf_filter.update(u[:,k], data["pr"][k], Q, R, dyn_func_params={"dt":dt}, meas_func_params={"sat_pos":sat_pos_k})

# Time horizon
N = 10
n = 5 # state is x = [x, y, z, b, bd]
m = 3

problem = nlp.fixedTimeOptimalEstimationNLP(N, T, n, m)

# Define variables
X = problem.addVariables(N+1, n, name='x')
xhat0 = np.vstack((EKF["x_ENU"].reshape(1,-1),
                   EKF["y_ENU"].reshape(1,-1),
                   EKF["z_ENU"].reshape(1,-1),
                   EKF["bias"].reshape(1,-1),
                   bias_rate_guess*np.ones((1,len(t)))))
problem.initializeEstimate(X, t, xhat0)

# Define system dynamics
U, W = problem.addDynamics(dynamics.gnss_pos_and_bias, X, t, u, )
problem.addDynamicsCost(cost_functions.weighted_l2_norm, W, {"Q":np.linalg.inv(Q)})

# Define cost function, adding measurements individually
for k in range(T+1):
    for (i, pr) in enumerate(data["pr"][k]):
        sat_pos_ENU = utils.ecef2enu(data["sat_pos"][k][i,:], p_ref_ECEF)
        params = {"sat_pos": sat_pos_ENU}
        R = np.diag([r_pr])
        y = np.array([[pr]])
        tk = np.array([[data["t"][k]]])
        problem.addResidualCost(measurements.pseudorange, X, tk, y, np.linalg.inv(R), params)

# Solve problem
print('Building problem.')
problem.build()

print('Solving problem.')
problem.solve()
problem.solve(warmstart=True)

NLP = {"t":t, "p_ref_ECEF":p_ref_ECEF, "bias":np.zeros(T+1),
      "x_ENU":np.zeros(T+1), "y_ENU":np.zeros(T+1), "z_ENU":np.zeros(T+1),
      "lat":np.zeros(T+1), "lon":np.zeros(T+1), "h":np.zeros(T+1)}
x_opt = problem.extractSolution('x', t)
for k in range(T+1):
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
plt.scatter(LS["x_ENU"], LS["y_ENU"], c='r', marker='x', label='LS')
plt.scatter(LS_batch["x_ENU"], LS_batch["y_ENU"], c='k', marker='s', label='Batch LS')
plt.scatter(EKF["x_ENU"], EKF["y_ENU"], c='g', marker='d', label='EKF')
plt.scatter(NLP["x_ENU"], NLP["y_ENU"], c='b', marker='o', label='NLP')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.legend()

plt.figure(2)
plt.plot(LS["t"], LS["x_ENU"], c='r', label='x (LS)')
plt.plot(LS_batch["t"], LS_batch["x_ENU"]*np.ones(T+1), c='k', label='x (Batch LS)')
plt.plot(EKF["t"], EKF["x_ENU"], c='g', label='x (EKF)')
plt.plot(NLP["t"], NLP["x_ENU"], c='b', label='x (NLP)')
plt.xlabel('t (s)')
plt.ylabel('x (m)')
plt.legend()

plt.figure(3)
plt.plot(LS["t"], LS["y_ENU"], c='r', label='y (LS)')
plt.plot(LS_batch["t"], LS_batch["y_ENU"]*np.ones(T+1), c='k', label='y (Batch LS)')
plt.plot(EKF["t"], EKF["y_ENU"], c='g', label='y (EKF)')
plt.plot(NLP["t"], NLP["y_ENU"], c='b', label='y (NLP)')
plt.xlabel('t (s)')
plt.ylabel('y (m)')
plt.legend()

plt.show()



