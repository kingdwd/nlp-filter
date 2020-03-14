import numpy as np
import gnss
import utils


def buildGeometryMatrix(sat_pos, x):
    """ Builds the geometry matrix with row equal to
    -(xsat - x)/norm(x - xsat) """
    N = sat_pos.shape[0]
    G = np.zeros((N, 3))

    for k in range(N):
        LoS = sat_pos[k,:] - x
        G[k,:] = -LoS/np.linalg.norm(LoS)

    return G


def iterativeLeastSquares(sat_pos, pr, x=np.zeros(3), b=0, maxiter=100):
    N = sat_pos.shape[0]
    for i in range(maxiter):
        # Build geometry matrix
        G = buildGeometryMatrix(sat_pos, x)
        G = np.hstack((G, np.ones((N, 1)))) # add column for bias term

        # Build residual vector
        drho = np.zeros(N)
        for k in range(N):
            drho[k] = pr[k] - np.linalg.norm(sat_pos[k,:] - x) - b

        # Solve least squares        
        dx = np.matmul(np.linalg.pinv(G), drho)

        # Update solution
        x += dx[0:3]
        b += dx[3]

        # Exit if converged
        if np.linalg.norm(dx) < 1e-7:
            break

    return x, b


def iterativeLeastSquaresVel(sat_pos, sat_vel, pr_rate, x):
    N = sat_pos.shape[0]

    # Build geometry matrix
    G = buildGeometryMatrix(sat_pos, x)
    G = np.hstack((G, np.ones((N, 1)))) # add column for bias term

    # Build residual vector
    drho = np.zeros(N)
    for k in range(N):
        LoS_dir = -G[k,:3] # normalized line of sight
        drho[k] = pr_rate[k] - np.dot(sat_vel[k,:], LoS_dir)

    # Solve least squares        
    sol = np.matmul(np.linalg.pinv(G), drho)
    v = sol[:3]
    bd = sol[3]

    return v, bd


def iterativeLeastSquares_multiTimeStep(t, sat_pos, pr, x=np.zeros(3), b0=0, alpha=0, maxiter=100):
    """ t is an array of times that is of the same length as sat_pos and pr and corresponds
    to the time at the point that measurement was taken """
    N = sat_pos.shape[0]
    for i in range(maxiter):
        # Build geometry matrix
        G = buildGeometryMatrix(sat_pos, x)
        G = np.hstack((G, np.ones((N, 1)))) # add column for bias term
        G = np.hstack((G, t.reshape(N,1))) # add column for alpha term

        # Build residual vector
        drho = np.zeros(N)
        for k in range(N):
            drho[k] = pr[k] - np.linalg.norm(sat_pos[k,:] - x) - b0 - alpha*t[k]

        # Solve least squares        
        dx = np.matmul(np.linalg.pinv(G), drho)

        # Update solution
        x += dx[0:3]
        b0 += dx[3]
        alpha += dx[4]

        # Exit if converged
        if np.linalg.norm(dx) < 1e-7:
            print('Multi time step least squares converged.')
            break

    return x, b0, alpha


def runLeastSquares(t, sat_pos, pr, sat_vel=None, pr_rate=None, p_ref_ECEF=None):
    """ Computes least squares position/velocity solution for each time step 
    individually """
    T = t.shape[0]
    sol = {"t":t, "p_ref_ECEF":p_ref_ECEF, "bias":np.zeros(T), "bias_rate":np.zeros(T),
      "x_ECEF":np.zeros(T), "y_ECEF":np.zeros(T), "z_ECEF":np.zeros(T),
      "xd_ECEF":np.zeros(T), "yd_ECEF":np.zeros(T), "zd_ECEF":np.zeros(T),
      "x_ENU":np.zeros(T), "y_ENU":np.zeros(T), "z_ENU":np.zeros(T),
      "xd_ENU":np.zeros(T), "yd_ENU":np.zeros(T), "zd_ENU":np.zeros(T),
      "lat":np.zeros(T), "lon":np.zeros(T), "h":np.zeros(T)}

    for t in range(T):
        # Solve least squares
        p_ECEF, b = iterativeLeastSquares(sat_pos[t], pr[t])
        p_LLA = utils.ecef2lla(p_ECEF)
        sol["x_ECEF"][t] = p_ECEF[0]
        sol["y_ECEF"][t] = p_ECEF[1]
        sol["z_ECEF"][t] = p_ECEF[2]
        sol["lat"][t] = p_LLA[0]
        sol["lon"][t] = p_LLA[1]
        sol["h"][t] = p_LLA[2]
        sol["bias"][t] = b

        # Convert to ENU coordinates
        if p_ref_ECEF is not None:
            p_ENU = utils.ecef2enu(p_ECEF, p_ref_ECEF)
            sol["x_ENU"][t] = p_ENU[0]
            sol["y_ENU"][t] = p_ENU[1]
            sol["z_ENU"][t] = p_ENU[2]

        # Compute rates as well
        if sat_vel is not None:
            pdot_ECEF, bdot = iterativeLeastSquaresVel(sat_pos[t], sat_vel[t], pr_rate[t], p_ECEF)
            sol["xd_ECEF"][t] = pdot_ECEF[0]
            sol["yd_ECEF"][t] = pdot_ECEF[1]
            sol["zd_ECEF"][t] = pdot_ECEF[2]
            sol["bias_rate"][t] = bdot

            if p_ref_ECEF is not None:
                pdot_ENU = utils.ecef2enu(pdot_ECEF, p_ref_ECEF, rotation_only=True)
                sol["xd_ENU"][t] = pdot_ENU[0]
                sol["yd_ENU"][t] = pdot_ENU[1]
                sol["zd_ENU"][t] = pdot_ENU[2]
        
    return sol


def runBatchLeastSquares(t, sat_pos, pr, p_ref_ECEF=None):
    """ Computes a batch least squares estimate under the assumption
    that the receiver is stationary and that the receiver clock bias
    is linearly changing. """
    sat_pos_batch = np.array([]).reshape(0,3)
    pr_batch = np.array([])
    t_batch = np.array([])
    for i in range(t.shape[0]):
        N_i = pr[i].shape[0]
        sat_pos_batch = np.vstack((sat_pos_batch, sat_pos[i]))
        pr_batch = np.hstack((pr_batch, pr[i]))
        t_batch = np.hstack((t_batch, [t[i]]*N_i))

    p_ECEF, b0, alpha = iterativeLeastSquares_multiTimeStep(t_batch, sat_pos_batch, pr_batch)
    p_LLA = utils.ecef2lla(p_ECEF)
    sol = {"t":t, "p_ref_ECEF":p_ref_ECEF,  "b0":b0, "alpha":alpha,
            "x_ECEF":p_ECEF[0], "y_ECEF":p_ECEF[1], "z_ECEF":p_ECEF[2],
            "lat":p_LLA[0], "lon":p_LLA[1], "h":p_LLA[2]}

    if p_ref_ECEF is not None:
        p_ENU = utils.ecef2enu(p_ECEF, p_ref_ECEF)
        sol["x_ENU"] = p_ENU[0]
        sol["y_ENU"] = p_ENU[1]
        sol["z_ENU"] = p_ENU[2]

    return sol
    