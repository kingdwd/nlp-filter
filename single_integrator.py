import numpy as np
import matplotlib.pyplot as plt
import utils.nlp as nlp
import utils.dynamics as dynamics
import utils.cost_functions as cost_functions
import utils.constraints as constraints
import utils.simulate as simulate
import utils.measurements as measurements
import pdb

# Time horizon
T = 10.
N = 30
n = 2
m = 2

problem = nlp.fixedTimeOptimalControlNLP(N, T, n, m)

# Define variables
X = problem.addVariables(N+1, 2, name='x')
U = problem.addVariables(N+1, 2, name='u')

# Define system dynamics
problem.addDynamics(dynamics.single_integrator, X, U)

# Define cost function
problem.addStageCost(cost_functions.single_integrator, X, U)

# Add initial condition constraint
x0 = [-3, 4]
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
plt.plot(t, u_opt[:,0], '-.', label='u0')
plt.plot(t, u_opt[:,1], '-.', label='u1')
plt.xlabel('t')
plt.legend()
plt.show()



