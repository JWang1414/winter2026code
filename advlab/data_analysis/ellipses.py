import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Settings
SAVE_PLOTS = False

# File paths
MEASUREMENT_PATH = "advlab/data_analysis_assignment/measurements.csv"
SAVE_LOCATION = "temp/"


def gaussian(x, a, mu, sigma):
    """
    Gaussian function
    """
    return a * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


def lognormal(x, a, mu, sigma):
    """
    Lognormal function
    """
    return a / (x * sigma) * np.exp(- (np.log(x) - mu) ** 2 / (2 * sigma ** 2))


def laplacian(x, a, mu, sigma):
    """
    Laplacian function
    """
    return a * np.exp(-np.abs(x - mu) / sigma)


# Import data from CSV file
data = pd.read_csv(MEASUREMENT_PATH)

# Sort the data
data = data.sort_values(by='x (units)')

# Extract x and y data
x_data = data['x (units)'].values
y_data = data['y (units)'].values

# Extract uncertainties
x_unc = data['x (units).1'].values
y_unc = data['y (units).1'].values

# Define possible functions to fit
possible_functions = {
    'Gaussian': gaussian,
    'Lognormal': lognormal,
    'Laplacian': laplacian
}

# Initial parameter guesses for each function
initial_guesses = {
    'Gaussian': [np.mean(y_data), np.mean(x_data), np.std(x_data)],
    'Lognormal': [np.mean(y_data), np.mean(np.log(x_data)), np.std(np.log(x_data))],
    'Laplacian': [np.mean(y_data), np.mean(x_data), np.std(x_data)]
}

# Pre-define storage for results
optimized_values = dict()
residual_list = dict()

# Go through each possible function and fit the data
for current_function in possible_functions.keys():
    func = possible_functions[current_function]
    p0 = initial_guesses[current_function]

    try:
        # Fit the function to the data
        popt, pcov = curve_fit(func, x_data, y_data, p0=p0, sigma=y_unc, absolute_sigma=True)
        
        # Calculate residuals
        fitted_function = func(x_data, *popt)
        residuals = y_data - fitted_function
        
        # Evaluate goodness of fit
        chi_squared = np.sum((residuals / y_unc) ** 2)
        dof = len(y_data) - len(popt)
        pcov = np.sqrt(np.diag(pcov))
        
        # Print results
        print(f"--- {current_function.upper()} ---")
        print(f"Optimal parameters: {popt}")
        print(f"Covariance matrix: {pcov}")
        print(f"Chi-squared: {chi_squared}")
        print(f"Reduced Chi-squared: {chi_squared / dof}\n")

        # Save the fitted function and residuals
        optimized_values[current_function] = popt
        residual_list[current_function] = residuals

    except Exception as e:
        print(f"Could not fit {current_function}: {e}\n")

# Plot the fitted functions
xx = np.linspace(min(x_data), max(x_data), 500)
for func, popt in optimized_values.items():
    plt.plot(xx, possible_functions[func](xx, *popt), label=f'{func}')

# Plot the original data on-top everything
plt.errorbar(x_data, y_data, xerr=x_unc, yerr=y_unc, fmt='o', label='Measurements')

# Labels
plt.xlabel('x (units)')
plt.ylabel('y (units)')
plt.title('Horizontal and Vertical Axes of Ellipses')
plt.legend()
plt.tight_layout()

# Save/show plots
if SAVE_PLOTS:
    plt.savefig(SAVE_LOCATION + "ellipse_fits_plot.png")
else:
    plt.show()

# Define the subplots for residuals
fig, axs = plt.subplots(len(residual_list), sharex=True)

# Plot residuals for each function
for i, func in enumerate(residual_list.keys()):
    residuals = residual_list[func]
    axs[i].errorbar(x_data, residuals, yerr=y_unc, fmt='o', label=func)

    # Implement an axis-line, title, and legend for each of the subplots
    axs[i].axhline(0, color='red', linestyle='--')
    axs[i].legend()

# Labels
fig.supxlabel('x (units)')
fig.supylabel('Residuals (units)')
fig.suptitle('Residuals of Fitted Functions')
fig.tight_layout()

# Save/show plots
if SAVE_PLOTS:
    plt.savefig(SAVE_LOCATION + "ellipse_residuals_plot.png")
else:
    plt.show()
