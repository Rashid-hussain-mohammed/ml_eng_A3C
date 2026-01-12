import numpy as np
import matplotlib.pyplot as plt

def objectives(x):
    m, P, Cd = x
    energy = 0.0005 * m + 0.02 * Cd * P
    accel = m / P
    return energy, accel

bounds = np.array([
    [800, 1800],
    [60, 200],
    [0.25, 0.40]
])

X = np.random.uniform(bounds[:,0], bounds[:,1], size=(500, 3))
F = np.array([objectives(x) for x in X])

plt.scatter(F[:,0], F[:,1])
plt.xlabel("Energy Consumption")
plt.ylabel("Acceleration Time")
plt.title("Random Design Space (Sanity Check)")
plt.show()
plt.savefig("results/pareto_front2.png", dpi=300)