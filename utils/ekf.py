import numpy as np


class EKF(object):
    """ Uses the notation from Prob. Robotics by 
    S. Thrun et al.
    Inputs:
    x_t, G_t = dyn_func(x_t-1, u_t) where G_t = dg/dx(x_t-1, u_t)
    z_t, H_t = meas_func(x_t) where H_t = dh/dx(x_t) """

    def __init__(self, dyn_func, meas_func, mu0, S0):
        # Initialize with prior estimate mu and covariance S
        self.mu = mu0
        self.S = S0

        # Define functions
        self.dynamics = dyn_func
        self.measurement = meas_func

    def update(self, u, z, Q, R, dyn_func_params=None, meas_func=None, meas_func_params=None):
        """ Given current time u_t and z_t, updates 
        the state and covariance estimate from the
        previous time step (t-1) to the current
        time step xhat_t, P_t. 
        Q is the covariance of the dynamics
        R is the covariance of the measurement. """

        # Predict state
        mu_pred, S_pred = self.predict(u, Q, dyn_func_params)

        # Update based on measurement if one is provided
        if z is not None:
            mu_cor, S_cor = self.correct(mu_pred, S_pred, z, R, meas_func, meas_func_params)
            self.mu = mu_pred + mu_cor
            self.S = S_pred + S_cor
        else:
            self.mu = mu_pred
            self.S = S_pred 

    def predict(self, u, Q, dyn_func_params):
        # Predict new state and covariance
        mu_pred, G = self.dynamics(self.mu, u, params=dyn_func_params, jac=True)
        S_pred = np.matmul(G, np.matmul(self.S, G.T)) + Q
        return mu_pred, S_pred

    def correct(self, mu_pred, S_pred, z, R, meas_func, meas_func_params):
        if meas_func is None:
            meas_func = self.measurement

        # Predict measurement
        z_pred, H = meas_func(mu_pred, params=meas_func_params, jac=True)

        # Compute gain
        P = np.matmul(H, np.matmul(S_pred, H.T)) + R
        K = np.matmul(S_pred, np.matmul(H.T, np.linalg.inv(P)))

        # Correct predictions based on measurements
        mu_cor = np.matmul(K, z - z_pred)
        S_cor = -np.matmul(K, np.matmul(H, S_pred))

        return mu_cor, S_cor 

