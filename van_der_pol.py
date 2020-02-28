import numpy as np
import matplotlib.pyplot as plt
import utils.nlp as nlp
import utils.dynamics as dynamics
import utils.cost_functions as cost_functions
import utils.constraints as constraints


# Time horizon
T = 10.
N = 20
n = 2
m = 1

problem = nlp.fixedTimeOptimalControlNLP(N, T, n, m)

# Define variables
x_lb = [-np.inf, -0.25]
x_ub = [np.inf, np.inf]
X = problem.addVariables(N+1, 2, x_lb, x_ub, 'x')

u_lb = [-1.0]
u_ub = [1.0]
U = problem.addVariables(N+1, 1, u_lb, u_ub, 'u')

# Define system dynamics
problem.addDynamics(dynamics.van_der_pol, X, U)

# Define cost function
problem.addStageCost(cost_functions.van_der_pol, X, U)

# Add initial condition constraint
x0 = [0, 1]
problem.addInitialCondition(X[0], x0)

# Solve problem
problem.build()
problem.solve()

t = np.linspace(0, T, 20)
x_opt = problem.extractSolution('x', t)
u_opt = problem.extractSolution('u', t)

plt.figure(1)
plt.plot(t, x_opt[:,0], '--', label='x0')
plt.plot(t, x_opt[:,1], '-', label='x1')
plt.plot(t, u_opt, '-.', label='u')
plt.xlabel('t')
plt.legend()
plt.show()



