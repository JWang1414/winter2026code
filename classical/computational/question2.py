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
t_span = (0, 1000)
t_points = np.linspace(t_span[0], t_span[1], int(1e4))
poincare_points = np.arange(t_span[0], t_span[1], 2 * np.pi)


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
poincare_y = sol.sol(poincare_points)

# Poincare Section
plt.plot(poincare_y[0], poincare_y[1], "o", markersize=1)
plt.xlabel("Position")
plt.ylabel("Velocity")
save_show_plots(f"poincare_section_{amp}")

# # Time series of position
# plt.plot(t_points, y[0], "-")
# plt.xlabel("Time")
# plt.ylabel("Position")
# save_show_plots("time_series")

# # Time series of velocity
# plt.plot(t_points, y[1], "-")
# plt.xlabel("Time")
# plt.ylabel("Velocity")
# save_show_plots("time_series_velocity")

# # Phase plot of the results
# plt.plot(y[0], y[1], "-")
# plt.xlabel("Position")
# plt.ylabel("Velocity")
# save_show_plots(f"phase_plot_driven_{amp}")
