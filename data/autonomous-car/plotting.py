from matplotlib import pyplot as plt
import pickle
import numpy as np
from scipy.interpolate import interp1d
import os
import pdb

def load_obj(fname):
    with open(fname + '.pkl', 'rb') as f:
        return pickle.load(f)


dir_path = os.path.dirname(os.path.realpath(__file__))
sim_path = os.path.join(dir_path, 'sim')
data_path = os.path.join(dir_path, 'filtering')
save_path = os.path.join(dir_path, 'figures')

traj = load_obj(sim_path + '/traj_data')
LS = load_obj(data_path + '/leastsquares')
EKF = load_obj(data_path + '/ekf')
NLP_L2 = load_obj(data_path + '/nlp-l2')
NLP_Huber = load_obj(data_path + '/nlp-huber')

# LS results
ls_t = []
ls_x = []
ls_y = []
for (i,t) in enumerate(LS["t"]):
    if i % 10 == 0:
        ls_x.append(LS["x_ENU"][i])
        ls_y.append(LS["y_ENU"][i])
        ls_t.append(LS["t"][i])
ls_x = np.array(ls_x)
ls_y = np.array(ls_y)
ls_t = np.array(ls_t)

# EKF results
ekf_t = []
ekf_x = []
ekf_y = []
for (i,t) in enumerate(EKF["t"]):
    if i % 10 == 0:
        ekf_x.append(EKF["x_ENU"][i])
        ekf_y.append(EKF["y_ENU"][i])
        ekf_t.append(EKF["t"][i])
ekf_x = np.array(ekf_x)
ekf_y = np.array(ekf_y)
ekf_t = np.array(ekf_t)

# NLP results with L2 norm
nlp_l2_x = []
nlp_l2_y = []
nlp_l2_t = []
i_sorted = np.argsort(NLP_L2["t"])
t = np.array(NLP_L2["t"])
x = np.array(NLP_L2["x_ENU"])
y = np.array(NLP_L2["y_ENU"])
t_sorted = t[i_sorted]
x_sorted = x[i_sorted]
y_sorted = y[i_sorted]
for (i,t) in enumerate(t_sorted):
    if i % 5 == 0:
        nlp_l2_x.append(x_sorted[i])
        nlp_l2_y.append(y_sorted[i])
        nlp_l2_t.append(t)
nlp_l2_x = np.array(nlp_l2_x)
nlp_l2_y = np.array(nlp_l2_y)
nlp_l2_t = np.array(nlp_l2_t)

# NLP results with huber loss
nlp_huber_x = []
nlp_huber_y = []
nlp_huber_t = []
i_sorted = np.argsort(NLP_Huber["t"])
t = np.array(NLP_Huber["t"])
x = np.array(NLP_Huber["x_ENU"])
y = np.array(NLP_Huber["y_ENU"])
t_sorted = t[i_sorted]
x_sorted = x[i_sorted]
y_sorted = y[i_sorted]
for (i,t) in enumerate(t_sorted):
    if i % 5 == 0:
        nlp_huber_x.append(x_sorted[i])
        nlp_huber_y.append(y_sorted[i])
        nlp_huber_t.append(t)
nlp_huber_x = np.array(nlp_huber_x)
nlp_huber_y = np.array(nlp_huber_y)
nlp_huber_t = np.array(nlp_huber_t)

# Compute cumulative errors
true_pos = np.vstack((traj["x"][0,:], traj["x"][1,:]))
true_t = traj["t"]
f = interp1d(true_t, true_pos)

error_ls = np.zeros(ls_t.shape)
for (i,t) in enumerate(ls_t):
    true = f(t)
    error_ls[i] = np.linalg.norm(true - np.array([ls_x[i], ls_y[i]]))
print('LS: Mean: {}, Max: {}, Std: {}'.format(np.mean(error_ls), np.max(error_ls), np.std(error_ls)))

error_ekf = np.zeros(ekf_t.shape)
for (i,t) in enumerate(ekf_t):
    true = f(t)
    error_ekf[i] = np.linalg.norm(true - np.array([ekf_x[i], ekf_y[i]]))
print('EKF: Mean: {}, Max: {}, Std: {}'.format(np.mean(error_ekf), np.max(error_ekf), np.std(error_ekf)))

error_nlp_l2 = np.zeros(nlp_l2_t.shape)
for (i,t) in enumerate(nlp_l2_t):
    true = f(t)
    error_nlp_l2[i] = np.linalg.norm(true - np.array([nlp_l2_x[i], nlp_l2_y[i]]))
print('NLP-L2: Mean: {}, Max: {}, Std: {}'.format(np.mean(error_nlp_l2), np.max(error_nlp_l2), np.std(error_nlp_l2)))

error_nlp_huber = np.zeros(nlp_huber_t.shape)
for (i,t) in enumerate(nlp_huber_t):
    true = f(t)
    error_nlp_huber[i] = np.linalg.norm(true - np.array([nlp_huber_x[i], nlp_huber_y[i]]))
print('NLP-Huber: Mean: {}, Max: {}, Std: {}'.format(np.mean(error_nlp_huber), np.max(error_nlp_huber), np.std(error_nlp_huber)))

# Computation times
print('NLP-L2 mean time: {}, NLP-Huber mean time: {}'.format(np.mean(NLP_L2["t_solve"]), np.mean(NLP_Huber["t_solve"])))

# Plotting
plt.figure(1)
plt.plot(traj["x"][0,:], traj["x"][1,:], c='k', label='Ground Truth')
plt.plot(LS["x_ENU"], LS["y_ENU"], c='m', label='Least Squares')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.legend()
plt.savefig(save_path + '/sim.png', format='png', dpi=1000)

plt.figure(2)
plt.plot(traj["t"], traj["x"][3,:], c='b', label='Longitudinal velocity')
plt.plot(traj["t"], traj["x"][4,:], c='r', label='Lateral velocity')
plt.xlabel('t (s)')
plt.ylabel('v (m/s)')
plt.legend()
plt.savefig(save_path + '/sim_velocity.png', format='png', dpi=1000)

plt.figure(3)
plt.plot(traj["x"][0,:], traj["x"][1,:], c='k', label='Ground Truth')
plt.plot(ekf_x, ekf_y, c='g', label='EKF')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.legend()
plt.savefig(save_path + '/ekf.png', format='png', dpi=1000)

plt.figure(4)
plt.plot(traj["x"][0,:], traj["x"][1,:], c='k', label='Ground Truth')
plt.plot(nlp_l2_x, nlp_l2_y, c='b', label='NLP, l2-loss')
plt.plot(nlp_huber_x, nlp_huber_y, c='r', label='NLP, Huber loss')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.legend()
plt.savefig(save_path + '/nlp.png', format='png', dpi=1000)

plt.show()