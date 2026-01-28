import matplotlib.pyplot as plt
import numpy as np

# Remember to change the name of the file handing it in

SHOW_PLOTS = True
SAVE_PATH = "temp/"


def cosine(t, a, omega, phi):
    """Compute a cosine function

    Args:
        t (array-like): Time values
        a (float): Amplitude
        omega (float): Angular frequency
        phi (float): Phase shift

    Returns:
        array-like: Cosine values
    """
    return a * np.cos(omega * t + phi)


def sine(t, a, omega, phi):
    """Compute a sine function

    Args:
        t (array-like): Time values
        a (float): Amplitude
        omega (float): Angular frequency
        phi (float): Phase shift

    Returns:
        array-like: Sine values
    """
    return a * np.sin(omega * t + phi)


# Define the linspace the system will be plot on
xx = np.linspace(0, 8 * np.pi, int(1e3))

# Define the cosine functions
x1 = cosine(xx, 3, 25 / 4, np.pi / 5)
x2 = cosine(xx, 4, 7, np.pi / 8)
x_sum = x1 + x2

# Define the derivatives
v1 = sine(xx, -75 / 4, 25 / 4, np.pi / 5)
v2 = sine(xx, -28, 7, np.pi / 8)
v_sum = v1 + v2

# Organize the position and velocities into lists
position_functions = {"x1": x1, "x2": x2, "x_sum": x_sum}
velocity_functions = {"v1": v1, "v2": v2, "v_sum": v_sum}

# Position as a function of time
fig, axs = plt.subplots(len(position_functions), 1, figsize=(8, 8), sharex=True)
for ax, function_name in zip(axs, position_functions.keys()):
    ax.plot(xx, position_functions[function_name], label=f"{function_name}")
    ax.legend(loc="lower right")
    ax.grid(True)

# Label the axes
fig.suptitle("Position as a function of time")
fig.supxlabel("Time")
fig.supylabel("Position")
fig.tight_layout()

# Save/show plots
if SHOW_PLOTS:
    plt.show()
else:
    plt.savefig(SAVE_PATH + "position.png")

# Velocity as a function of time
fig, axs = plt.subplots(len(velocity_functions), 1, figsize=(8, 8), sharex=True)
for ax, function_name in zip(axs, velocity_functions.keys()):
    ax.plot(xx, velocity_functions[function_name], label=f"{function_name}")
    ax.legend(loc="lower right")
    ax.grid(True)

# Label the axes
fig.suptitle("Velocity as a function of time")
fig.supxlabel("Time")
fig.supylabel("Velocity")
fig.tight_layout()

# Save/show plots
if SHOW_PLOTS:
    plt.show()
else:
    plt.savefig(SAVE_PATH + "velocity.png")

# Phase space
names_list = list(position_functions.keys())
index = 0
fig, axs = plt.subplots(len(position_functions), 1, figsize=(8, 8), sharex=True)

for ax, position_func, velo_func in zip(
    axs, position_functions.values(), velocity_functions.values()
):
    ax.plot(position_func, velo_func, label=f"{names_list[index]}")
    ax.legend(loc="lower right")
    ax.grid(True)
    index += 1

# Label the axes
fig.suptitle("Phase space")
fig.supxlabel("Position")
fig.supylabel("Velocity")
fig.tight_layout()

# Save/show plots
if SHOW_PLOTS:
    plt.show()
else:
    plt.savefig(SAVE_PATH + "phase.png")
