import numpy as np
import pdb


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
            break

    return x, b0, alpha


def buildGeometryMatrix(sat_pos, x):
    """ Builds the geometry matrix with row equal to
    -(xsat - x)/norm(x - xsat) """
    N = sat_pos.shape[0]
    G = np.zeros((N, 3))

    for k in range(N):
        xsat = np.array([sat_pos])
        LoS = sat_pos[k,:] - x
        G[k,:] = -LoS/np.linalg.norm(LoS)

    return G


def pseudorange(x, params=None, jac=False):
    """ x = [x, y, z, b, bd] where b is receiver clock
    bias that is assumed to be linearly changing at rate
    bd """
    y = np.sqrt((x[0] - params["sat_pos"][0])**2 +
                    (x[1] - params["sat_pos"][1])**2 +
                    (x[2] - params["sat_pos"][2])**2) + \
                    x[3]
    if jac:
        J = np.zeros(5)

        # Line of sight vector
        LoS = params["sat_pos"] - x[:3]
        J[:3] = -LoS/np.linalg.norm(LoS)
        J[3] = 1.0
        return y, J
    else:
        return y


def multi_pseudorange(x, params=None, jac=False):
    """ x = [x, y, z, b, bd] where b is receiver clock
    bias that is assumed to be linearly changing at rate
    bd.
    Output is an array of psuedoranges """

    N = params["sat_pos"].shape[0]
    y = np.zeros(N)
    if jac:
        J = np.zeros((N,5))

    for i in range(N):
        if jac:
            y[i], J[i,:] = pseudorange(x, params={"sat_pos": params["sat_pos"][i,:]}, jac=True)
        else:
            y[i] = pseudorange(x, params={"sat_pos": params["sat_pos"][i,:]})

    if jac:
        return y, J
    else:
        return y


def multi_pseudorange_and_bias(x, params=None, jac=False):
    """ x = [x, y, z, b, bd] where b is receiver clock
    bias that is assumed to be linearly changing at rate
    bd.
    Output is an array of psuedoranges and bias """
    N = params["sat_pos"].shape[0]
    y = np.zeros(N+1)
    J = np.zeros((N+1, 5))
    y[-1] = x[3] # bias
    if jac:
        y[:-1], J[:-1,:] = multi_pseudorange(x, params={"sat_pos": params["sat_pos"]}, jac=True)
        return y, J
    else:
        y[:-1], J[:-1,:] = multi_pseudorange(x, params={"sat_pos": params["sat_pos"]}, jac=False)
        return y


def gnss_pos_and_bias(x, u, params=None, jac=False):
    """ x = [x, y, z, b, bd] where b is receiver clock
    bias that is assumed to be linearly changing
    Dynamics: x+ = x + dt*u and b+ = b + dt*bd
    """
    x += params["dt"]*np.array([u[0], u[1], u[2], x[4], 0.0])

    if jac:
        J = np.eye(5)
        J[3, 4] = params["dt"]
        return x, J
    else:
        return x




        
        