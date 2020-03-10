import numpy as np
import pickle

Ts = []
Ns = []
F = []
delta = []

# Phase 1, go straight for a bit
Ts.append(5)
Ns.append(100*Ts[0])
F.append(np.zeros(Ns[0]))
delta.append(np.zeros(Ns[0]))

# Phase 2, ease into a turn
Ts.append(2)
Ns.append(100*Ts[1])
F.append(100*np.linspace(0, 1, Ns[1]))
delta.append(np.deg2rad(15)*np.linspace(0, 1, Ns[1]))

# Phase 3, constant turn
Ts.append(1)
Ns.append(100*Ts[2])
F.append(100*np.ones(Ns[2]))
delta.append(np.deg2rad(15)*np.ones(Ns[2]))


# Phase 4, go straighten up
Ts.append(2)
Ns.append(100*Ts[3])
F.append(100*np.linspace(1, 50, Ns[3]))
delta.append(np.deg2rad(15)*np.linspace(1, 0, Ns[3]))

# Phase 5, accelerate
Ts.append(5)
Ns.append(100*Ts[4])
F.append(5000*np.ones(Ns[4]))
delta.append(np.zeros(Ns[4]))

# Phase 6 remove acceleration and start to turn
Ts.append(2)
Ns.append(100*Ts[5])
F.append(5000*np.linspace(1, 0, Ns[5]))
delta.append(np.deg2rad(-10)*np.linspace(0, 1, Ns[5]))

# Phase 7 turn
Ts.append(3)
Ns.append(100*Ts[6])
F.append(np.zeros(Ns[6]))
delta.append(np.deg2rad(-10)*np.ones(Ns[6]))

# Phase 8 end turn
Ts.append(1)
Ns.append(100*Ts[7])
F.append(500*np.linspace(0, 1, Ns[7]))
delta.append(np.deg2rad(-10)*np.linspace(1, 0, Ns[7]))

# Phase 9 go straight for a while
Ts.append(10)
Ns.append(100*Ts[8])
F.append(500*np.linspace(1, 2, Ns[8]))
delta.append(np.zeros(Ns[8]))

# Phase 10, ease into a turn
Ts.append(2)
Ns.append(100*Ts[9])
F.append(1000*np.ones(Ns[9]))
delta.append(np.deg2rad(-9)*np.linspace(0, 1, Ns[9]))

# Phase 11, constant turn
Ts.append(4)
Ns.append(100*Ts[10])
F.append(1000*np.ones(Ns[10]))
delta.append(np.deg2rad(-9)*np.ones(Ns[10]))

# Phase 12, go straighten up
Ts.append(2)
Ns.append(100*Ts[11])
F.append(1000*np.linspace(1, 0, Ns[11]))
delta.append(np.deg2rad(-9)*np.linspace(1, 0, Ns[11]))

# Phase 13 go straight for a while
Ts.append(8)
Ns.append(100*Ts[12])
F.append(3000*np.linspace(0, 1, Ns[12]))
delta.append(np.zeros(Ns[12]))

# Phase 14, hard into a turn
Ts.append(1)
Ns.append(100*Ts[13])
F.append(3000*np.ones(Ns[13]))
delta.append(np.deg2rad(35)*np.linspace(0, 1, Ns[13]))

# Phase 15, constant turn
Ts.append(5)
Ns.append(100*Ts[14])
F.append(3000*np.ones(Ns[14]))
delta.append(np.deg2rad(35)*np.ones(Ns[14]))

# Phase 16, go straighten up
Ts.append(1)
Ns.append(100*Ts[15])
F.append(3000*np.linspace(1, 1.75, Ns[15]))
delta.append(np.deg2rad(35)*np.linspace(1, 0, Ns[15]))

# Phase 17, go straight
Ts.append(1)
Ns.append(100*Ts[16])
F.append(5250*np.linspace(1, 0, Ns[16]))
delta.append(np.zeros(Ns[16]))

# Phase 18, go straight
Ts.append(5)
Ns.append(100*Ts[17])
F.append(0*np.linspace(1, 0, Ns[17]))
delta.append(np.zeros(Ns[17]))

# Phase 19, ease into a turn
Ts.append(5)
Ns.append(100*Ts[18])
F.append(1000*np.linspace(0, 1, Ns[18]))
delta.append(np.deg2rad(-15)*np.linspace(0, 1, Ns[18]))

# Phase 20, ease out of turn
Ts.append(15)
Ns.append(100*Ts[19])
F.append(1000*np.ones(Ns[19]))
delta.append(np.deg2rad(-15)*np.linspace(1, 0, Ns[19]))

# Phase 21, ease into a turn
Ts.append(2)
Ns.append(100*Ts[20])
F.append(1000*np.linspace(1, 0, Ns[20]))
delta.append(np.deg2rad(-8.8)*np.linspace(0, 1, Ns[20]))

# Phase 22, ease out of turn
Ts.append(2)
Ns.append(100*Ts[21])
F.append(np.zeros(Ns[21]))
delta.append(np.deg2rad(-8.8)*np.linspace(1, 0, Ns[21]))

# Phase 23, go straight
Ts.append(7)
Ns.append(100*Ts[22])
F.append(np.zeros(Ns[22]))
delta.append(np.zeros(Ns[22]))


u = np.array([]).reshape((2,-1))
T = 0
N = 0
for i in range(len(F)):
    ui = np.vstack((F[i], delta[i]))
    u = np.hstack((u, ui))
    T += Ts[i]
    N += Ns[i]
ui = np.vstack((F[-1][-1], delta[-1][-1]))
u = np.hstack((u, ui))

x0 = np.zeros(6)
x0[3] = 10.
x0[4] = 0.

t = np.linspace(0, T, N+1)
dt = t[1] - t[0]

sim_def = {"u":u, "T":T, "N":N, "x0":x0, "t":t, "dt":dt}
with open('sim_definition.pkl', 'wb') as f:
    pickle.dump(sim_def, f, pickle.HIGHEST_PROTOCOL)