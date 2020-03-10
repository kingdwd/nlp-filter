import os
import numpy as np
import csv
import pdb
from scipy.interpolate import interp1d
import pickle

# Use pylog to convert data to csv format
# fname = 'log_164_2020-2-27-10-03-56'
fname = 'log_165_2020-2-27-10-05-46'
os.system('ulog2csv ' + fname + '.ulg')

t1 = []
throttle = []
steer = []
with open(fname + '_manual_control_setpoint_0.csv') as f:
    data = csv.reader(f)
    for (i,row) in enumerate(data):
        if i >= 1:
            t1.append(float(row[0]))
            throttle.append(float(row[3]))
            steer.append(float(row[4]))

t2 = []
angular_vel_x = []
angular_vel_y = []
angular_vel_z = []
acc_x = []
acc_y = []
acc_z = []
with open(fname + '_sensor_combined_0.csv') as f:
    data = csv.reader(f)
    for (i,row) in enumerate(data):
        if i >= 1:
            t2.append(float(row[0]))
            angular_vel_x.append(float(row[1]))
            angular_vel_y.append(float(row[2]))
            angular_vel_z.append(float(row[3]))
            acc_x.append(float(row[6]))
            acc_y.append(float(row[7]))
            acc_z.append(float(row[8]))

# Delete unneeded csv files
os.system('rm ./' + fname + '_*.csv')

# Convert time to seconds and zero out
t1 = np.array(t1)*10**-6
t2 = np.array(t2)*10**-6
t0 = np.min([t1[0], t2[0]])
t1 -= t0
t2 -= t0

# Convert to numpy arrays and interpolate control to higher rate
control = np.vstack((throttle, steer)) # [0,1]
interp = interp1d(t1, control, fill_value="extrapolate")
control = interp(t2)

# Angular rate in rad/s
ang_rate = np.vstack((angular_vel_x, angular_vel_y, angular_vel_z))

# Acceleration in m/s^2
acc = np.vstack((acc_x, acc_y, acc_z))

data = {"t":t2, "u":control, "ang_rate":ang_rate, "acc":acc}

# Save data
with open(fname + '.pkl', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)



