from casadi import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import pdb

class ChebyshevPseudospectralMethod():
    def __init__(self, N, t0, tf):
        # super(ChebyshevPseudospectralCollocation, self).__init__()
        self.N = N
        self.t0 = t0
        self.tf = tf

        # Generate nodes
        self.buildNodes() # stores nodes in self.tau

        # Generate differentiation matrix
        self.buildDiffMatrix() # stores matrix as self.D

        # Generate weights to approximate integral via quadrature
        self.buildQuadratureWeights()

        # Compute the Lagrange polynomials phi_j for j = 0 to N
        self.buildLagrangePolynomials()

    def tau2t(self, tau):
        """ Converts from the interval [-1,1] to the interval [t0,tf] """
        return 0.5*((self.tf-self.t0)*tau + (self.tf + self.t0))

    def t2tau(self, t):
        """ Converts from the interval [t0,tf] to the interval [-1,1] """
        return (2.0*t - (self.tf-self.t0))/(self.tf-self.t0)

    def buildNodes(self):
        """ Returns N+1 CGL (Chebyshev-Gauss-Lobatto) time points on domain [-1,1] for the extrema
        of the Nth order Chebyshev polynomial. """
        tau = np.zeros(self.N+1)
        for k in range(self.N+1):
            tau[k] = np.cos(k*np.pi/self.N)
        self.tau = tau[::-1] # standard formulation does [1,-1] so reorder

    def buildDiffMatrix(self):
        """ Returns differentiation matrix D, such that xdot(tau_k) = sum D_kj*x_j,
        where x_j are the column state vector at the node point tau_j. This is based
        on the selection of the Lagrange interpolating polynomials with the
        Chebyshev-Gauss-Lobatto points. """
        CGL_nodes = self.tau[::-1] # need to flip them back to [1, -1] interval for D computation
        D = np.zeros((self.N+1, self.N+1))
        for k in range(self.N+1):
            c = np.ones(self.N+1)
            c[0] = 2
            c[self.N] = 2
            for j in range(self.N+1):
                if k == 0 and j == 0:
                    D[k,j] = (2*self.N**2 + 1)/6.0
                elif k == self.N and j == self.N:
                    D[k,j] = -(2*self.N**2 + 1)/6.0
                elif k == j:
                    D[k,j] = -CGL_nodes[k]/(2*(1-CGL_nodes[k]**2))
                else:
                    D[k,j] = (c[k]/c[j])*(np.power(-1, j+k)/(CGL_nodes[k] - CGL_nodes[j]))

        # Sort to have the time vector [-1,1] instead of [1,-1]
        self.D = -D

    def buildQuadratureWeights(self):
        """ Computes the weights for discretizing integral based on Clenshaw-Curtis
        quadrature scheme """
        w = np.zeros(self.N + 1)
        if self.N % 2 == 0:
            w[0] = 1.0/(self.N**2 - 1)
            w[self.N] = w[0]
            a = 0
        else:
            w[0] = 1.0/self.N**2
            w[self.N] = w[0]
            a = 1
        for s in range(1,(self.N-a)/2 + 1):
            w[s] = 2.0/self.N
            for j in range(1,(self.N-a)/2):
                w[s] += (4.0/self.N)*(1.0/(1.0-4.0*j**2))*np.cos(2*np.pi*j*s/self.N)
                w[s] += (2.0/self.N)*(1.0/(1-(self.N-a)**2))*np.cos((self.N-a)*s*np.pi/self.N)
                w[self.N-s] = w[s]

        self.w = w

    def buildLagrangePolynomials(self):
        """ Computes the Lagrange polynomial basis functions phi(t)
        Inputs:
        t: CGL nodes """
        phi = []
        for j in range(self.N+1):
            # Construct jth Lagrange polynomial
            phi_j = np.poly1d([1])
            for k in range(self.N+1):
                if k != j:
                    phi_j *= (1/(self.tau[j]-self.tau[k]))*np.poly1d([1, -self.tau[k]])
            phi.append(phi_j)

        self.phi = phi

    def evaluateLagrangePolynomials(self, t):
        """ With the approximation x(tau) = sum x_j phi_j(tau), this function
        computes the quantities phi_j(tau).
        Inputs t is the time quantity that is then automatically converted
        to the normalized tau quantity. """
        tau = self.t2tau(t)
        phi = np.zeros(self.N + 1)
        for j in range(self.N + 1):
            phi[j] = self.phi[j](tau)
        return phi

    def evaluateSolution(self, t, X):
        """ Computes x(t) = sum x_j phi_j(t).
        Input t is a time point (automatically converted to normalized tau)
        Input X is a list containing the vectors x_j in order """
        phi = self.evaluateLagrangePolynomials(t)
        x = np.zeros(X[0].shape)
        for j in range(self.N + 1):
            x += X[j]*phi[j]
        return x



