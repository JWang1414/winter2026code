import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = "12"

# Global variables
SAVE_FIGURE = True


def LJpotential(r):
    """Function that calculates the LJ potential"""
    return np.power(r, -12) - 2 * np.power(r, -6)


def LJforce(r):
    """Function that calculates the LJ force"""
    return 12 * np.power(r, -13) * (1 - np.power(r, 6))


# ----- FORCE AND POTENTIAL -----

rr = np.linspace(0.8, 3.0, 100)

fig, ax = plt.subplots(2, sharex=True)

ax[0].plot(rr, LJpotential(rr))
ax[0].set_ylabel("$V(r)$")
ax[0].grid()

ax[1].plot(rr, LJforce(rr))
ax[1].set_xlabel("$r$")
ax[1].set_ylabel("$F(r)$")
ax[1].grid()

fig.suptitle("Lennard-Jones Potential and Force")

if SAVE_FIGURE:
    plt.savefig("lj_potential_force.png")
else:
    plt.tight_layout()
    plt.show()

# ----- MOTION OF A PARTICLE IN THE LJ POTENTIAL -----

dt = 0.01  # time step.

end_time = 10.0  # length of simulation in atomic time units
time = np.arange(0, end_time, dt)  # define time array
nt = len(time)  # number of time steps

# %% Initializing
# empty arrays for the values of r and v.
radi = np.zeros(nt)  # array of zeros that we will fill below with radii
vels = np.zeros(nt)  # array of zeros that we will fill below with velocities

# Initial conditions
radi[0] = 1.02
vels[0] = 0.0

# Integrate using the Euler-Cromer method
for ii in range(nt - 1):
    vels[ii + 1] = vels[ii] + LJforce(radi[ii]) * dt
    radi[ii + 1] = radi[ii] + vels[ii + 1] * dt

# ----- DETERMINE PERIOD OF OSCILLATION -----

crossings = []
copy = radi.copy() - 1

# Find crossings where the particle changes sign across zero
for ii in range(nt - 1):
    if copy[ii + 1] * copy[ii] < 0:
        crossings.append(time[ii + 1])

# Calculate the period as twice the mean time between crossings
period = 2 * np.mean(np.diff(crossings))

print(f"Period: {period:.3f}")

# ----- Plotting -----

plt.figure()

# Position plot
plt.subplot(2, 1, 1)
plt.plot(time, radi)

plt.ylabel("$r(t)/r_m$")
plt.grid()

# Velocity plot
plt.subplot(2, 1, 2)
plt.plot(time, vels)

plt.xlabel(r"$t$ (units of $r_m\sqrt{m/\epsilon}$)")
plt.ylabel(r"$v(t)$ (units of $\sqrt{\epsilon/m}$)")
plt.grid()

# Add a title over both plots
plt.suptitle("$r_0 = {0:.2f}r_m$".format(radi[0]))
plt.tight_layout()

# Save/show figure
if SAVE_FIGURE:
    plt.savefig("LJ_r0={0:.2f}.png".format(radi[0]))
else:
    plt.show()
