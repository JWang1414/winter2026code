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
accel_raw = pd.read_csv("acceleration.csv")
no_sandpaper_raw = pd.read_csv("nopaper.csv")
sandpaper_raw = pd.read_csv("paper.csv")

# Organize data into dictionaries
acceleration_data = {
    "domino": accel_raw["Average"].to_numpy(),
    "domino_unc": accel_raw["Average.1"].to_numpy(),
    "speed": accel_raw["Speed"].to_numpy(),
    "speed_unc": accel_raw["Speed.1"].to_numpy(),
    "1/C": accel_raw["1/C"].to_numpy(),
    "1/C_unc": accel_raw["1/C.1"].to_numpy(),
}

nopaper_data = {
    "spacing": no_sandpaper_raw["Spacing"].to_numpy(),
    "spacing_unc": no_sandpaper_raw["Spacing.1"].to_numpy(),
    "speed": no_sandpaper_raw["Speed"].to_numpy(),
    "speed_unc": no_sandpaper_raw["Speed.1"].to_numpy(),
    "1/C": no_sandpaper_raw["1/C"].to_numpy(),
    "1/C_unc": no_sandpaper_raw["1/C.1"].to_numpy(),
}

paper_data = {
    "spacing": sandpaper_raw["Spacing"].to_numpy(),
    "spacing_unc": sandpaper_raw["Spacing.1"].to_numpy(),
    "speed": sandpaper_raw["Speed"].to_numpy(),
    "speed_unc": sandpaper_raw["Speed.1"].to_numpy(),
    "1/C": sandpaper_raw["1/C"].to_numpy(),
    "1/C_unc": sandpaper_raw["1/C.1"].to_numpy(),
}

# Setup plotting parameters in dictionaries
acceleration_speed = {
    "x": acceleration_data["domino"],
    "xerr": acceleration_data["domino_unc"],
    "y": acceleration_data["speed"],
    "yerr": acceleration_data["speed_unc"],
    "title": "Average Speed vs Domino",
    "xlabel": "Domino Number",
    "ylabel": "Speed (cm/s)",
}

acceleration_c = {
    "x": acceleration_data["domino"],
    "xerr": acceleration_data["domino_unc"],
    "y": acceleration_data["1/C"],
    "yerr": acceleration_data["1/C_unc"],
    "title": "Calculated 1/C for each Domino",
    "xlabel": "Domino Number",
    "ylabel": "1/C",
}

nopaper_speed = {
    "x": nopaper_data["spacing"],
    "xerr": nopaper_data["spacing_unc"],
    "y": nopaper_data["speed"],
    "yerr": nopaper_data["speed_unc"],
    "title": "Average Terminal Velocity vs Spacing (w/o Sandpaper)",
    "xlabel": "Spacing (Multiples of Domino Thickness)",
    "ylabel": "Speed (cm/s)",
}

nopaper_c = {
    "x": nopaper_data["spacing"],
    "xerr": nopaper_data["spacing_unc"],
    "y": nopaper_data["1/C"],
    "yerr": nopaper_data["1/C_unc"],
    "title": "Calculated 1/C for each Spacing (w/o Sandpaper)",
    "xlabel": "Spacing (Multiples of Domino Thickness)",
    "ylabel": "1/C",
}

paper_speed = {
    "x": paper_data["spacing"],
    "xerr": paper_data["spacing_unc"],
    "y": paper_data["speed"],
    "yerr": paper_data["speed_unc"],
    "title": "Average Terminal Velocity vs Spacing (w/Sandpaper)",
    "xlabel": "Spacing (Multiples of Domino Thickness)",
    "ylabel": "Speed (cm/s)",
}

paper_c = {
    "x": paper_data["spacing"],
    "xerr": paper_data["spacing_unc"],
    "y": paper_data["1/C"],
    "yerr": paper_data["1/C_unc"],
    "title": "Calculated 1/C for each Spacing (w/Sandpaper)",
    "xlabel": "Spacing (Multiples of Domino Thickness)",
    "ylabel": "1/C",
}


def plot_data(plot_params):
    # Skip plotting if flag is set
    if SKIP_PLOTS:
        return

    # Plot data
    plt.errorbar(
        plot_params["x"],
        plot_params["y"],
        xerr=plot_params["xerr"],
        yerr=plot_params["yerr"],
        fmt="o",
    )

    # Labels
    plt.xlabel(plot_params["xlabel"])
    plt.ylabel(plot_params["ylabel"])
    plt.grid()
    plt.tight_layout()

    # Save or show plot
    if SAVE_PLOTS:
        filename = plot_params["title"].replace(" ", "_").replace("/", "-")
        plt.savefig(f"{SAVE_PATH}{filename}.png")
        plt.clf()
    else:
        plt.show()


if __name__ == "__main__":
    # Generate plots
    plot_data(acceleration_speed)
    plot_data(acceleration_c)
    plot_data(nopaper_speed)
    plot_data(nopaper_c)
    plot_data(paper_speed)
    plot_data(paper_c)

    # Extract the average 1/C from acceleration data
    avg_1_C = np.mean(acceleration_data["1/C"][2:])
    print(f"Average 1/C from acceleration data: {avg_1_C}")
