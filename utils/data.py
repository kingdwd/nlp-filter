import numpy as np
from scipy.io import loadmat
import pickle

C = 299792458 # speed of light in m/s


def load_gnss_logs(prefix):
    """ Loads prefix + 'satposecef.mat' and
    prefix + 'ranges.mat' and perform the correction
    for the ionosphere """

    sat_data = loadmat(prefix + 'satposecef.mat')
    all_sat_pos = sat_data["svPoss"][1:,:,:3] # ECEF coordinates (meters)
    ion_correction = sat_data["svPoss"][1:,:,3] # meters
    sat_clock_bias = sat_data["svPoss"][1:,:,4] # seconds

    # Load data and correct for ionosphere and satellite clock bias
    pr_data = loadmat(prefix + 'ranges.mat')
    if pr_data["pseudoranges"].ndim == 2:
        all_pr = pr_data["pseudoranges"][1:,:] + ion_correction + C*sat_clock_bias
        sats = pr_data["pseudoranges"][0,:] # satellite SVID
        pos_only = True
        print('Assuming data at 1 Hz')
        times = range(all_pr.shape[0])

    elif pr_data["pseudoranges"].ndim == 3:
        all_pr = pr_data["pseudoranges"][1:,:,0] + ion_correction + C*sat_clock_bias
        all_pr_rate = pr_data["pseudoranges"][1:,:,1]
        all_sat_vel = pr_data["pseudoranges"][1:,:,2:5]
        times = np.max(pr_data["pseudoranges"][1:,:,5], axis=1)
        sats = pr_data["pseudoranges"][0,:,0] # satellite SVID
        pos_only = False

    # Parse the data by time step and remove all invalid data (NaN)
    T, N = all_pr.shape
    sat_pos = []
    sat_vel = []
    pr = []
    pr_rate = []
    for t in range(T):
        # Only take measurements that are not NaN
        sat_pos_t = np.array([]).reshape(0,3)
        sat_vel_t = np.array([]).reshape(0,3)
        pr_t = np.array([])
        pr_rate_t = np.array([])
        for i in range(N):
            if not np.all(all_sat_pos[t,i,:] == 0.0) and not np.isnan(all_pr[t,i]):
                if pos_only:
                    sat_pos_t = np.vstack((sat_pos_t, all_sat_pos[t,i,:].reshape((1,3))))
                    pr_t = np.hstack((pr_t, all_pr[t,i]))
                else:
                    sat_pos_t = np.vstack((sat_pos_t, all_sat_pos[t,i,:].reshape((1,3))))
                    sat_vel_t = np.vstack((sat_vel_t, all_sat_vel[t,i,:].reshape((1,3))))
                    pr_t = np.hstack((pr_t, all_pr[t,i]))
                    pr_rate_t = np.hstack((pr_rate_t, all_pr_rate[t,i]))

        # Append data from time t to list
        sat_pos.append(sat_pos_t)
        sat_vel.append(sat_vel_t)
        pr.append(pr_t)
        pr_rate.append(pr_rate_t)

    # Store data in dictionary
    data = {"t":times, "sats":sats, "sat_pos":sat_pos, "pr":pr}
    if not pos_only:
        data["sat_vel"] = sat_vel
        data["pr_rate"] = pr_rate

    return data


def load_px4_logs(prefix):
    return load_obj(prefix)


def save_obj(obj, fname):
    with open(fname + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(fname):
    with open(fname + '.pkl', 'rb') as f:
        return pickle.load(f)