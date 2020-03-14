from matplotlib import pyplot as plt
import pickle
import numpy as np
from scipy.interpolate import interp1d
import os
import csv
import pdb

def load_obj(fname):
    with open(fname + '.pkl', 'rb') as f:
        return pickle.load(f)


dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(dir_path, 'filtering')
save_path = os.path.join(dir_path, 'figures')

LS_A = load_obj(data_path + '/leastsquaresA')
LS_B = load_obj(data_path + '/leastsquaresB')
# EKF = load_obj(data_path + '/ekf')
NLP = load_obj(data_path + '/nlp')

# Compute distances
d = np.zeros(len(NLP["t"]))
for (i, t) in enumerate(NLP["t"]):
    pA = np.array([NLP["xA_ENU"][i], NLP["yA_ENU"][i]])
    pB = np.array([NLP["xB_ENU"][i], NLP["yB_ENU"][i]])
    d[i] = np.linalg.norm(pA-pB)

with open('LS_A.csv', mode='w') as f:
    writer = csv.writer(f, delimiter=',')
    for i in range(len(LS_A["t"])):
        writer.writerow([LS_A["lat"][i], LS_A["lon"][i]])

with open('LS_B.csv', mode='w') as f:
    writer = csv.writer(f, delimiter=',')
    for i in range(len(LS_B["t"])):
        writer.writerow([LS_B["lat"][i], LS_B["lon"][i]])

with open('NLP_A.csv', mode='w') as f:
    writer = csv.writer(f, delimiter=',')
    for i in range(len(NLP["t"])):
        writer.writerow([NLP["latA"][i], NLP["lonA"][i]])

with open('NLP_B.csv', mode='w') as f:
    writer = csv.writer(f, delimiter=',')
    for i in range(len(NLP["t"])):
        writer.writerow([NLP["latB"][i], NLP["lonB"][i]])

# Computation times
print('NLP mean time: {}'.format(np.mean(NLP["t_solve"])))

# Plotting
plt.figure(1)
plt.scatter(LS_A["x_ENU"][10:], LS_A["y_ENU"][10:], c='m', label='LS, Rec. A', s=1)
plt.scatter(LS_B["x_ENU"][10:], LS_B["y_ENU"][10:], c='m', label='LS, Rec. B', s=1)
plt.plot(NLP["xA_ENU"], NLP["yA_ENU"], c='k', label='NLP, Rec. A')
plt.plot(NLP["xB_ENU"], NLP["yB_ENU"], c='k', label='NLP, Rec. B')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.legend()
plt.axis('equal')
plt.savefig(save_path + '/comparison.png', format='png', dpi=1000)

plt.show()

