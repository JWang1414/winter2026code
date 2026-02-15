import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import pi

# Global variables
SAVE_FIGURE = True
SAVE_PATH = 'images/'

# Plot settings
plt.rcParams["figure.dpi"] = 150  # for crisper figures
plt.rcParams["font.size"] = "16"  # change the font size

# Physical parameters
drive_period = 2 * pi  # driving period 2*pi
omega = np.arange(1, 128, 2)  # driving angular frequency
omega0 = 1.5 * np.ones(len(omega))  # natural angular frequency
q_factor = 10.0  # Q-factor
gamma = omega0 / q_factor  # damping inverse time scale

time = np.linspace(0.0, 3 * drive_period, 1024)  # from t=0 to t=3T in 1024 points

# Compute the Fourier series approximation of the square wave function
a_n = [4/(n * np.pi) * (-1)**((n-1)/2) for n in omega]
cosines_forcing = [np.cos(n * time) for n in omega]

forcing = np.dot(a_n, cosines_forcing)

# Amplitude of the response
pp = omega0**2 - omega**2
qq = 2 * gamma * omega
amplitude = omega0**2 / (pp**2 + qq**2) ** 0.5

# Phase of the response
pp_safe = np.where(abs(pp) < 1e-14, 1e-14, pp)  # to avoid division by zero
delta = np.arctan(qq / pp_safe)

# Adjust phase convention
    # Numpy convention: -pi/2<=arctan<=pi/2
    # Lecture convention: 0<=arctan<=pi => need adjusting
delta = np.where(delta < 0, delta + pi, delta)

# Compute the result
cosines_solution = [np.cos(n * time - delta[i]) for i, n in enumerate(omega)]
xsol = np.dot(amplitude, cosines_solution)

# Generate the plot
plt.plot(time / drive_period, xsol, label="response")
plt.plot(time / drive_period, forcing, "r--", label="forcing")
plt.xlim(time[0] / drive_period, time[-1] / drive_period)

plt.xlabel(r"$t/T$")
plt.ylabel("$x(t)$")
plt.title("Response of a Harmonic Oscillator to a Square Wave Forcing")
plt.legend()
plt.tight_layout()
plt.grid()

if SAVE_FIGURE:
    plt.savefig(SAVE_PATH + 'square_wave_response.png')
    plt.clf()
else:
    plt.show()
