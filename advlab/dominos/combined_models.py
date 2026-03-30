import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Change the font size
plt.rcParams.update({"font.size": 14})

# Global variables
SAVE_PLOTS = True
SAVE_PATH = "images/"
SKIP_PLOTS = False

# Import data
no_sandpaper_raw = pd.read_csv("nopaper.csv")
sandpaper_raw = pd.read_csv("paper.csv")
banks_raw = pd.read_csv("banks_results.csv")
leeuwen_raw = pd.read_csv("vanleeuwen_results.csv")

# Process the data
nopaper_data = {
    "x": no_sandpaper_raw["Spacing"].to_numpy(),
    "xerr": no_sandpaper_raw["Spacing.1"].to_numpy(),
    "y": no_sandpaper_raw["Speed"].to_numpy(),
    "yerr": no_sandpaper_raw["Speed.1"].to_numpy(),
}

paper_data = {
    "x": sandpaper_raw["Spacing"].to_numpy(),
    "xerr": sandpaper_raw["Spacing.1"].to_numpy(),
    "y": sandpaper_raw["Speed"].to_numpy(),
    "yerr": sandpaper_raw["Speed.1"].to_numpy(),
}

# Convert to the correct units
for data in [nopaper_data, paper_data]:
    x_scale = 7.5 / 48
    y_scale = 1 / np.sqrt(4.8 * 9.8 * 100)
    data["x"] = np.multiply(data["x"], x_scale)
    data["xerr"] = np.multiply(data["xerr"], x_scale)
    data["y"] = np.multiply(data["y"], y_scale)
    data["yerr"] = np.multiply(data["yerr"], y_scale)

# Plot the two models
plt.plot(banks_raw.iloc[:, 0], banks_raw.iloc[:, 1], label="Banks")
plt.plot(leeuwen_raw.iloc[:, 0], leeuwen_raw.iloc[:, 1], label="Van Leeuwen")

# Plot the data
plt.errorbar(
    nopaper_data["x"],
    nopaper_data["y"],
    xerr=nopaper_data["xerr"],
    yerr=nopaper_data["yerr"],
    label="No Sandpaper",
)
plt.errorbar(
    paper_data["x"],
    paper_data["y"],
    xerr=paper_data["xerr"],
    yerr=paper_data["yerr"],
    label="Sandpaper",
)

plt.xlabel("Separation Ratio")
plt.ylabel("Dimensionless Propagation Speed")
plt.grid()
plt.legend()
plt.tight_layout()

if SAVE_PLOTS:
    plt.savefig(SAVE_PATH + "combined_models.png")
if not SKIP_PLOTS:
    plt.show()
