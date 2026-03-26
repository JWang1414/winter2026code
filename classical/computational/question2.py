import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

# Global variables
SAVE_PLOTS = False
SAVE_LOCATION = "images/q2/"

# Define physical constants
y0 = (2.0, 0.0, 0.0)
alpha = 1.0
beta = -1.0
gamma = 0.3
omega = 1.5
amp = 0.6

# Define simulation parameters
t_span = (0, 500)
t_points = np.linspace(t_span[0], t_span[1], int(1e4))


def duffing(t, y):
    x, v, phi = y
    dxdt = v
    dphidt = omega
    dvdt = -gamma * v - beta * x - alpha * x**3 + amp * np.cos(phi)
    return [dxdt, dvdt, dphidt]


def save_show_plots(name=""):
    if SAVE_PLOTS and name:
        plt.savefig(f"{SAVE_LOCATION}{name}.png")
        plt.clf()
    elif SAVE_PLOTS:
        raise ValueError("name must be provided when SAVE_PLOTS is True")
    else:
        plt.show()


# Solve the ODE
sol = solve_ivp(duffing, t_span, y0, dense_output=True)

# Use the continuous solution for plotting
y = sol.sol(t_points)

# Phase plot of the results
plt.plot(y[0], y[1], "-")
plt.xlabel("Position")
plt.ylabel("Velocity")
save_show_plots("phase_plot2")
