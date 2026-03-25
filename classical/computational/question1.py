import matplotlib.pyplot as plt
import numpy as np


def phi_dot(theta):
    return -np.sin(theta)


def theta_dot(phi):
    return phi


def euler_cromer_step(theta, phi, dt):
    phi_new = phi + phi_dot(theta) * dt
    theta_new = theta + theta_dot(phi_new) * dt
    return theta_new, phi_new


def forward_euler_step(theta, phi, dt):
    theta_new = theta + theta_dot(phi) * dt
    phi_new = phi + phi_dot(theta_new) * dt
    return theta_new, phi_new


# Define the initial conditions
theta = [np.pi / 4]
phi = [0]

# Define the simulation parameters
current_function = forward_euler_step
total_time = 60
step_size = 0.05
num_steps = int(total_time / step_size)
time_array = np.arange(0, total_time + step_size, step_size)

# Run the simulation
for _ in range(num_steps):
    theta_new, phi_new = current_function(theta[-1], phi[-1], step_size)
    theta.append(theta_new)
    phi.append(phi_new)

# Compute the energy in the system
energy = [(1 - np.cos(theta[i])) + 0.5 * phi[i] ** 2 for i in range(num_steps + 1)]

# Plot the angle as a function of time
plt.plot(time_array, theta)

plt.xlabel("Time")
plt.ylabel("Theta")
plt.show()

# Plot the energy as a function of time
plt.plot(time_array, energy)

plt.xlabel("Time")
plt.ylabel("Energy")
plt.show()
