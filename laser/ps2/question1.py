import numpy as np
import matplotlib.pyplot as plt

# Define global variables
SAVE_PLOTS = False
SAVE_PATH = 'temp/'

# Define relevat physical constants
c = 3e8
density = 1e16
wavelength = 600e-9
a_21 = 2e8
z = 0.02
nu_0 = c / wavelength
fwhm = 1e9
hwhm = fwhm / 2

def lineshape_function(nu):
    delta_nu = nu - nu_0
    return (1 / np.pi) * (hwhm / (delta_nu**2 + 1 * hwhm**2))

def cross_section(nu):
    g_nu = lineshape_function(nu)
    return (wavelength**2 / (8 * np.pi)) * a_21 * g_nu

def transmitted_fraction(nu):
    sigma_nu = cross_section(nu)
    return np.exp(-density * sigma_nu * z)

# Frequency range around the central frequency
nu_range = np.linspace(nu_0 - 5*fwhm, nu_0 + 5*fwhm, 1000)
transmission = transmitted_fraction(nu_range)

# Find where the half-maximum occurs
half_max = np.mean([np.min(transmission), 1])

# Plotting the results
plt.plot((nu_range - nu_0) / 1e9, transmission, label='Transmitted Intensity')
plt.axhline(half_max, color='red', linestyle='--', label='Half Maximum')

plt.title('Transmitted Intensity of Light through Vapour Cell')
plt.xlabel('Frequency Detuning (GHz)')
plt.ylabel('Transmitted Fraction')
plt.legend()
plt.grid()

if SAVE_PLOTS:
    plt.savefig(SAVE_PATH + f"transmitted_intensity_{density:.1e}.png")
    plt.clf()
else:
    plt.show()

# Calculate and print the FWHM of the transmitted fraction
indices = np.where(transmission <= half_max)[0]
if len(indices) >= 2:
    fwhm_transmitted = (nu_range[indices[-1]] - nu_range[indices[0]]) / 1e9
    print(f"FWHM of transmitted fraction: {fwhm_transmitted:.2f} GHz")

### Saved output for density = 1e16, 1e17
# FWHM of transmitted fraction: 1.09 GHz
# FWHM of transmitted fraction: 2.11 GHz
