import matplotlib.pyplot as plt
import numpy as np

# Configuration
SAVE_FILES = True
SAVE_PATH = "temp/"
SKIP_NUMERICAL = True
DEBUG = False

# Physical constants
mass = 0.15  # [kg]
v0 = 20  # initial speed [m/s]
grav = 9.81  # gravitational acceleration [m/s2]
c = 1.225e-3  # drag coefficient

vter = np.sqrt(mass * grav / c)
ymax = 0.5 * vter**2 / grav * np.log(1 + v0**2 / vter**2)
vground = -vter * v0 * (vter**2 + v0**2) ** -0.5  # velocity when hitting ground [m/s]

if DEBUG:
    print("c = {0:.2e} N.s2/m2, vter = {1:.2e} m/s".format(c, vter))
    print("ymax = {0:.2e} m".format(ymax))
    print("When hitting ground, v = {0:.2e} m/s".format(vground))

# Analytical formulas for v = fct(y)
tmax = vter * np.arctan(v0 / vter) / grav  # time to reach max height
tground = tmax - vter / grav * np.arctanh(vground / vter)  # time to hit ground

# 128 instants going from 0 to tmax (upward)
t_up = np.linspace(0.0, tmax, 128)
# same from tmax to tground (downward)
t_down = np.linspace(tmax, tground, 128)

v_up = vter * np.tan(np.arctan(v0 / vter) - grav * t_up / vter)
v_down = vter * np.tanh(grav * (tmax - t_down) / vter)

# numerical integration

errors = []  # empty list where we will record the errors
dt_values = []  # empty list where we will record the time steps

# Compute forward Euler with a different number of time steps
for nsteps in range(5, 51, 3):
    time = np.linspace(0.0, tground, nsteps)  # time array
    dt = time[1] - time[0]  # time step
    dt_values.append(dt)
    v_num = 0 * time  # initializing velocity array for forward Euler
    v_num[0] = v0

    # Compute forward Euler
    for ii in range(1, nsteps):
        if v_num[ii - 1] > 0:
            # Ball is moving up
            v_num[ii] = v_num[ii - 1] - dt * grav * (1 + (v_num[ii - 1] / vter) ** 2)
        else:
            # Ball is moving down
            v_num[ii] = v_num[ii - 1] - dt * grav * (1 - (v_num[ii - 1] / vter) ** 2)

    # calculate global error
    errors.append(abs(v_down[-1] - v_num[-1]))  # add value to list of errors

    # Skip the plot if SKIP_NUMERICAL is True
    if SKIP_NUMERICAL:
        continue

    # Plot the numerical solution
    plt.plot(t_up, v_up, "r", label="analytical")
    plt.plot(t_down, v_down, "r")
    plt.plot(time, v_num, "+:", label="numerical")

    # Labels
    plt.title("$v(t)$ for $dt={0:.4f}$ s ({1} time steps)".format(dt, nsteps))
    plt.xlabel("$t$ (s)")
    plt.ylabel("$v$ (m/s)")
    plt.legend()
    plt.tight_layout()
    plt.grid()

    # Save/show plots
    if SAVE_FILES:
        plt.savefig(SAVE_PATH + "friction_{}nsteps.png".format(nsteps))
        plt.clf()
    else:
        plt.show()

# Compare the global error with the time step
plt.plot(dt_values, errors, "o-")

# Labels
plt.xlabel("$dt$ (s)")
plt.ylabel("Global error (m/s)")
plt.tight_layout()
plt.grid()

# Save/show plots
if SAVE_FILES:
    plt.savefig(SAVE_PATH + "errors.png")
    plt.clf()
else:
    plt.show()
