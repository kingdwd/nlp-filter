import numpy as np
from scipy.integrate import ode
from scipy.interpolate import interp1d


def dynamics_wrapper(t, x, func, u, params=None):
    # u = np.zeros(2)
    return func(x, u(t), params)


def open_loop_sim(t, u, x0, func, params=None):
    """ Simulates an open-loop trajectory starting from x0
    and applying control u (m by T) array, with dynamics described
    by func """

    # Create interpolation function for control
    u_t = interp1d(t, u, fill_value="extrapolate")

    # Initialize the ode solver
    r = ode(dynamics_wrapper)
    r.set_initial_value(x0, t[0])
    r.set_f_params(func, u_t, params)

    # Simulate
    i = 0
    x = np.zeros((x0.shape[0], t.shape[0]))
    x[:,0] = x0
    while r.successful() and r.t < t[-1]:
        dt = t[i+1] - t[i]
        i += 1
        x[:,i] = r.integrate(r.t + dt)

    return x


def generate_measurements(x, measurement_model, sigma, params=None):
    """ Given state trajectory history x (n by T), the
    measurement model function measurement_model, and covariance matrix
    sigma for the measurements,
    returns a Gaussian random measurement history """
    T = x.shape[1]
    p = sigma.shape[0]

    y = np.zeros((p, T))
    for i in range(T):
        y[:,i] = measurement_model(x[:,i], params) + np.random.multivariate_normal(np.zeros(p), sigma)

    return y