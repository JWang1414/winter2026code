import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# TODO:
# Try to fit the curve(?). It disagrees with data
# If I have a lot of time, make this code nicer

# Global variables
SAVE_PLOTS = False
SAVE_PATH = 'images/'
SKIP_PLOTS = False

# Import data
accel_raw = pd.read_csv('advlab/dominos/acceleration.csv')
no_sandpaper_raw = pd.read_csv('advlab/dominos/nopaper.csv')
sandpaper_raw = pd.read_csv('advlab/dominos/paper.csv')

# Organize data into dictionaries
acceleration_data = {
    "domino": accel_raw['Average'].to_numpy(),
    "speed": accel_raw['Speed'].to_numpy(),
    "speed_unc": accel_raw['Speed.1'].to_numpy(),
    "1/C": accel_raw['1/C'].to_numpy(),
    "1/C_unc": accel_raw['1/C.1'].to_numpy()
}

nopaper_data = {
    "spacing": no_sandpaper_raw['Spacing'].to_numpy(),
    "speed": no_sandpaper_raw['Speed'].to_numpy(),
    "speed_unc": no_sandpaper_raw['Speed.1'].to_numpy(),
    "1/C": no_sandpaper_raw['1/C'].to_numpy(),
    "1/C_unc": no_sandpaper_raw['1/C.1'].to_numpy()
}

paper_data = {
    "spacing": sandpaper_raw['Spacing'].to_numpy(),
    "speed": sandpaper_raw['Speed'].to_numpy(),
    "speed_unc": sandpaper_raw['Speed.1'].to_numpy(),
    "1/C": sandpaper_raw['1/C'].to_numpy(),
    "1/C_unc": sandpaper_raw['1/C.1'].to_numpy()
}

# Setup plotting parameters in dictionaries
acceleration_speed = {
    "x": acceleration_data["domino"],
    "y": acceleration_data["speed"],
    "yerr": acceleration_data["speed_unc"],
    "title": "Average Speed vs Domino",
    "xlabel": "Domino Number",
    "ylabel": "Speed (cm/s)",
}

acceleration_c = {
    "x": acceleration_data["domino"],
    "y": acceleration_data["1/C"],
    "yerr": acceleration_data["1/C_unc"],
    "title": "Calculated 1/C for each Domino",
    "xlabel": "1/C",
    "ylabel": "Speed (cm/s)",
}

nopaper_speed = {
    "x": nopaper_data["spacing"],
    "y": nopaper_data["speed"],
    "yerr": nopaper_data["speed_unc"],
    "title": "Average Terminal Velocity vs Spacing (w/o Sandpaper)",
    "xlabel": "Spacing (Multiples of Domino Thickness)",
    "ylabel": "Speed (cm/s)",
}

nopaper_c = {
    "x": nopaper_data["spacing"],
    "y": nopaper_data["1/C"],
    "yerr": nopaper_data["1/C_unc"],
    "title": "Calculated 1/C for each Spacing (w/o Sandpaper)",
    "xlabel": "1/C",
    "ylabel": "Speed (cm/s)",
}

paper_speed = {
    "x": paper_data["spacing"],
    "y": paper_data["speed"],
    "yerr": paper_data["speed_unc"],
    "title": "Average Terminal Velocity vs Spacing (w/Sandpaper)",
    "xlabel": "Spacing (Multiples of Domino Thickness)",
    "ylabel": "Speed (cm/s)",
}

paper_c = {
    "x": paper_data["spacing"],
    "y": paper_data["1/C"],
    "yerr": paper_data["1/C_unc"],
    "title": "Calculated 1/C for each Spacing (w/Sandpaper)",
    "xlabel": "1/C",
    "ylabel": "Speed (cm/s)",
}

def plot_data(plot_params):
    # Skip plotting if flag is set
    if SKIP_PLOTS:
        return

    # Plot data
    plt.errorbar(plot_params["x"], plot_params["y"],
                 yerr=plot_params["yerr"],
                 fmt='o')

    # Labels
    plt.title(plot_params["title"])
    plt.xlabel(plot_params["xlabel"])
    plt.ylabel(plot_params["ylabel"])
    plt.grid()
    plt.tight_layout()

    # Save or show plot
    if SAVE_PLOTS:
        plt.savefig(f"{SAVE_PATH}{plot_params['title'].replace(' ', '_')}.png")
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
