import casadi
import numpy as np
from scipy.interpolate import interp1d
import pdb
import collocation
import constraints

class NLP(object):
    """NLP class"""
    def __init__(self, N):
        self.N = N
        self.var_names = [] # array holding variable names in order generated
        self.var_sizes = [] # array holding variable dimensions
        self.w = {} # the decision variables
        self.opti = casadi.Opti()
        self.sol = None

    def addVariables(self, N_var, n_var, lb=None, ub=None, name='x'):
        """ Builds N_var symbolic vectors of dimension n,
        names the vectors 'name_j'. Returns python list
        of the vectors. Parameters lb and ub are vectors of length
        n_var that lower and upper bound the variables"""
        X = []
        for i in range(N_var):
            var_name = name + '_' + str(i)
            self.var_names.append(var_name)
            self.var_sizes.append(n_var)
            x = self.opti.variable(n_var)
            self.w[var_name] = x
            X.append(x)

            if lb is not None:
                self.opti.subject_to(x >= lb)
            if ub is not None:
                self.opti.subject_to(x <= ub)

        return X

    def addParameter(self, N_var, n_var, val=None):
        P = []
        for i in range(N_var):
            p = self.opti.parameter(n_var)
            P.append(p)

            if val is not None:
                self.setParameter(p, val)

        return P

    def setParameter(self, p, val):
        self.opti.set_value(p, val)

    def build(self, verbose=True):
        self.opti.minimize(self.J)

        # Create an NLP solver
        p_opts = {"ipopt.print_level":0}
        s_opts = {}
        self.opti.solver("ipopt", p_opts, s_opts)

    def solve(self, w0=None, warmstart=False):
        if warmstart and self.sol is not None:
            print('Warmstarting with previous solution')
            self.opti.set_initial(self.sol.value_variables())

        elif w0 is not None:
            print('Using provided initial guesses')
            print('Still needs to be implemented')

        # Solve the NLP
        self.sol = self.opti.solve()
        self.solver = self.sol.stats()

    def extractVariableValue(self, name, idx=0):
        """ From solver solution, extract the value of the decision
        variable given by 'name_idx' """
        var_name = name + '_' + str(idx)
        if var_name in self.w.keys():
            if self.sol is not None:
                return np.array(self.sol.value(self.w[var_name])).reshape(-1)
            else:
                print('Need to run solve() first')
                return None
        else:
            print('Could not find ' + name + '_' + str(idx) + '.')
            return None

    def extractSolution(self, name, t_array):
        """ From solver solution, extracts all variables with 'name'
        and evaluates them at the times in t_array. If the variable
        'name' is of dimension n and t_array is dimension T then the output
        is dimension T by n. """

        # First get the discrete variables associate with name
        X = []
        for i in range(self.N + 1):
            value = self.extractVariableValue(name, i)
            if value is not None:
                X.append(value)
            else:
                return None

        T = len(t_array)
        sol = np.zeros((T, X[0].shape[0]))
        for t in range(T):
            sol[t,:] = self.CPM.evaluateSolution(t_array[t], X)

        return sol


class fixedTimeOptimalControlNLP(NLP):
    def __init__(self, N, T, n, m):
        super(fixedTimeOptimalControlNLP, self).__init__(N)
        self.T = T
        self.n = n
        self.m = m

        # Initialize the psuedospectral method
        self.CPM = collocation.ChebyshevPseudospectralMethod(self.N, 0, T)

        # Initialize cost
        self.J = 0

    def addDynamics(self, func, X, U, params=None):
        """ Adds dynamics constraints to the NLP where the dynamics are defined
        in the func function and the state dimension is n and control
        dimension is m. """
        if len(X) != self.N + 1 or len(U) != self.N + 1:
            print('X and U must have N+1 points defined.')

        # Define a Casadi function
        x = casadi.MX.sym('x', self.n)
        u = casadi.MX.sym('u', self.m)
        if params is not None:
            fdynamics = casadi.Function('f', [x, u], [func(x, u, params)])
        else:
            fdynamics = casadi.Function('f', [x, u], [func(x, u)])

        # Add dynamics constraints at each collocation point
        for k in range(self.N + 1):
            # Compute state derivative at the k-th collocation point
            d_k = 0
            for j in range(self.N + 1):
                d_k += self.CPM.D[k, j]*X[j]

            # Append collocation equations to constraint dynamics
            self.opti.subject_to(fdynamics(X[k], U[k]) == (2.0/self.T)*d_k)
            
    def addStageCost(self, func, X, U, params=None):
        # Define Casadi function
        x = casadi.MX.sym('x', self.n)
        u = casadi.MX.sym('u', self.m)
        lcost = casadi.Function('l', [x, u], [func(x, u, params)])

        # Define stage cost
        for k in range(self.N+1):
            self.J += (self.T/2.0)*lcost(X[k], U[k])*self.CPM.w[k]

    def addSingleConstraint(self, func, variables, params=None):
        """ Adds a constraint g(vars) = 0
        vars is a list of Casadi variables, in order to pass to func 
        func must output a single scalar quantity g(vars) = 0! """
        self.opti.subject_to(func(variables, params) == 0)

    def addInitialCondition(self, x0, values):
        """ Sets the initial condition
        x0 is the first state variable (since CPM method has both endpoints in collocation nodes)
        value is the vector initial condition value to set """
        self.opti.subject_to(x0 == values)

    def addTerminalCondition(self, xT, values):
        """ Sets the terminal condition
        xT is the last state variable (since CPM method has both endpoints in collocation nodes)
        value is the vector initial condition value to set """
        self.opti.subject_to(xT == values)


class fixedTimeOptimalEstimationNLP(NLP):
    def __init__(self, N, T, n, m):
        super(fixedTimeOptimalEstimationNLP, self).__init__(N)
        self.T = T
        self.n = n
        self.m = m

        # Initialize the psuedospectral method
        self.CPM = collocation.ChebyshevPseudospectralMethod(self.N, 0, T)

        # Initialize cost
        self.J = 0

    def addDynamics(self, func, X, t_array, u_array, Q, params=None):
        """ Adds dynamics constraints to the NLP where the dynamics are defined
        in the func function and the state dimension is n and control
        dimension is m. The argument u_array is the array of dimension (m by *) of the controls
        that were applied to the system from time 0 to T """
        if len(X) != self.N + 1:
            print('X must have N+1 points defined.')

        # Define a Casadi function
        xvar = casadi.MX.sym('x', self.n)
        uvar = casadi.MX.sym('u', self.m)
        fdynamics = casadi.Function('f', [xvar, uvar], [func(xvar, uvar, params)])

        # Define interpolation function for the control u
        u_t = interp1d(t_array, u_array, fill_value="extrapolate")

        # Define process noise variables
        W = self.addVariables(self.N + 1, self.n, name='w')
        
        # Add dynamics constraints at each collocation point
        for k in range(self.N + 1):
            # Compute state derivative at the k-th collocation point
            d_k = 0
            for j in range(self.N + 1):
                d_k += self.CPM.D[k, j]*X[j]
            # Append collocation equations to constraint dynamics
            self.opti.subject_to(W[k] + fdynamics(X[k], u_t(self.CPM.tau2t(self.CPM.tau[k]))) == (2.0/self.T)*d_k)

            # Weight process noise variables with w^T*Q*w
            self.J += casadi.mtimes(W[k].T, casadi.mtimes(Q, W[k]))

    def addResidualCost(self, measurement_model, X, t_array, y_array, R, params=None):
        p = y_array.shape[0]

        # Define Casadi function
        x = casadi.MX.sym('x', self.n)
        y = casadi.MX.sym('y', p)
        residual = casadi.Function('l', [x, y], [y - measurement_model(x, params)])

        # Define stage cost
        for (i, t) in enumerate(t_array):
            # Build the expression for x at time t based on
            phi_t = self.CPM.evaluateLagrangePolynomials(t)
            X_t = 0
            for j in range(self.N + 1):
                X_t += X[j]*phi_t[j]

            # Then add to the cost
            r_t = residual(X_t, y_array[:,i])
            self.J += casadi.mtimes(r_t.T, casadi.mtimes(R, r_t))