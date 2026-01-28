import matplotlib.pyplot as plt
import numpy as np

# Define universal constants here
c = 3 * 10**8
h = 4.136 * 10**(-15)
k = 8.617 * 10**(-5)

def spectral_density(v, T):
    """
    Given the frequency v and temperature T,
    plot the spectral density resulting from blackbody radiation
    """
    term1 = 8 * np.pi * v**2
    term2 = h * v
    term3 = c**3
    term4 = np.exp(h*v / k*T) - 1
    top = term1 * term2
    bottom = term3 * term4
    return top / bottom

# Define simulation constants here
cmb = {
    "range": 10**11,
    "temperature": 2.7,
    "title": "Spectral Energy Density at CMB Temperature"
}

room = {
    "range": 10**9,
    "temperature": 298.15,
    "title": "Spectral Energy Density at Room Temperature"
}

sun = {
    "range": 0.6 * 10**8,
    "temperature": 5800,
    "title": "Spectral Energy Density of the Sun"
}

current = dict(cmb)

# Define linspace range
vv = np.linspace(0.1, current["range"], 10**6)

# Compute the spectral density for each case
p = spectral_density(vv, current["temperature"])

# Compute 1/3 of the max spectral density
# I will use this to determine the bandwidth
max_energy_density = np.max(p)
bandwidth_condition = max_energy_density / 3

# Subtract this quantity from the spectral density distribution,
# and then use sign swaps to find the zeroes
p_copy = p - bandwidth_condition

zeroes = []
for i in range(1, len(p_copy) - 1):
    if p_copy[i] * p_copy[i - 1] < 0:
        zeroes.append(np.mean([vv[i], vv[i-1]]))

# Compute the bandwidth and max wavelength
bandwidth = zeroes[-1] - zeroes[0]
max_index, = np.where(p == max_energy_density)
max_wavelength, = c / vv[max_index]

# Print max wavelength and bandwidth
print(current["title"])
print(f"Max Wavelength: {max_wavelength:.2e}")
print(f"Bandwidth: {bandwidth:.2e}")
print(f"Coherence time: {1 / bandwidth :.2e}")

# Plot the data
for zero in zeroes:
    plt.axvline(zero, 0.1, 0.7, color="red", linestyle='--', label="Bandwidth marker")
plt.plot(vv, p, label="ρ(ν)")
plt.title(current["title"])
plt.xlabel("Frequency, ν (Hz)")
plt.ylabel("Spectral Energy Density, ρ (J/m^3-Hz)")
plt.legend()
plt.tight_layout()
# plt.savefig(f"./temp_images/{current["title"]}")
plt.show()

### Copy and pasted output
# Spectral Energy Density at CMB Temperature
# Max Wavelength: 1.38e-02
# Bandwidth: 4.15e+10
# Coherence time: 2.41e-11

# Spectral Energy Density at Room Temperature
# Max Wavelength: 1.52e+00
# Bandwidth: 3.76e+08
# Coherence time: 2.66e-09

# Spectral Energy Density of the Sun
# Max Wavelength: 2.96e+01
# Bandwidth: 1.93e+07
# Coherence time: 5.18e-08
