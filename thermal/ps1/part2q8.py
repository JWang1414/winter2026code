import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

SAVE_FILES = False

def gaussian(x, mu, sigma):
    """
    Computes a Gaussian distribution in the typical probability density form.
    """
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# Load the saved samples
samples_A = np.load('temp_images/samples_A.npy')
samples_B = np.load('temp_images/samples_B.npy')

# Combine samples into a list for processing
samples_list = [np.array(samples_A), np.array(samples_B)]

# Plot the histograms and fitted curves
plt.figure(figsize=(12, 5))
colors = ['blue', 'orange']
labels = ['Species A', 'Species B']

for i, samples in enumerate(samples_list):
    # Compute histogram bins
    counts, bin_edges = np.histogram(samples, bins=30, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Normalize the histogram
    counts = counts / np.sum(counts * np.diff(bin_edges))

    # Plot histogram
    plt.subplot(1, 2, i + 1)
    plt.hist(samples, bins=30, density=True, alpha=0.6, color=colors[i], label='Histogram')

    # Fit Gaussian
    mu_init = np.mean(samples)
    sigma_init = np.std(samples)
    popt, pcov = curve_fit(gaussian, bin_centers, counts, p0=[mu_init, sigma_init])
    mu_fit, sigma_fit = popt

    # Compute the optimized Gaussian curve
    x_fit = np.linspace(min(samples), max(samples), 200)
    y_fit = gaussian(x_fit, mu_fit, sigma_fit)

    # Plot the fitted curve
    plt.plot(x_fit, y_fit, color='red', linewidth=2, label='Fitted Gaussian')

    # Determine the quality of the fit
    pcov = np.sqrt(np.diag(pcov))
    residuals = counts - gaussian(bin_centers, *popt)
    chi_squared = np.sum((residuals ** 2) / gaussian(bin_centers, *popt))
    dof = len(bin_centers) - len(popt)

    # Print fit results
    print(f'Fit results for {labels[i]}: mu = {mu_fit:.4e}, sigma = {sigma_fit:.4e}')
    print(f"Chi-squared = {chi_squared:.4e}, dof = {dof:.4f}, chi-squared/dof = {chi_squared/dof:.4e}")
    print(f"Parameter uncertainties: mu = {pcov[0]:.4e}, sigma = {pcov[1]:.4e}")
    print()

    # Labels and title
    plt.title(f'Velocity Distribution of {labels[i]}')
    plt.xlabel('Velocity')
    plt.ylabel('Density')
    plt.legend()
    plt.grid()

# Adjust layout and show/save the plot
plt.tight_layout()

if SAVE_FILES:
    plt.savefig('temp_images/final_velocity_distributions.png')
else:
    plt.show()

### Printed output from one sample:
# Fit results for Species A: mu = 6.3566e-03, sigma = 3.1884e+00
# Chi-squared = 3.4775e-04, dof = 28, chi-squared/dof = 1.2420e-05
# Parameter uncertainties: mu = 7.2917e-03, sigma = 5.9537e-03

# Fit results for Species B: mu = 6.5215e-03, sigma = 2.2485e+00
# Chi-squared = 4.0919e-04, dof = 28, chi-squared/dof = 1.4614e-05
# Parameter uncertainties: mu = 5.3264e-03, sigma = 4.3490e-03
