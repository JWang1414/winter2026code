import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la

plt.rcParams["figure.dpi"] = 150  # for crisper figures
# plt.rcParams['font.size'] = '8'


def mass_matrix(Nmasses, masses):
    """Creates a Nmasses x Nmasses mass matrix for Nmasses masses
    masses is a 1D numpy array containing Nmasses mass values
    """
    M = np.zeros((Nmasses, Nmasses))
    for n in range(Nmasses):
        M[n, n] = masses[n]

    return M


def coupling_matrix(Nmasses, stiffnesses):
    """Creates the matrix of the spring forces
    stiffnesses is a 1D numpy array contatining Nmasses+1 stiffness values
    """
    K = np.zeros((Nmasses, Nmasses))
    for n in range(Nmasses):  # we go line-by-line
        K[n, n] = stiffnesses[n] + stiffnesses[n + 1]
        if n < Nmasses - 1:  # upper diagonal
            K[n, n + 1] = -stiffnesses[n + 1]
        if n > 0:  # lower diagonal
            K[n, n - 1] = -stiffnesses[n]

    return K


def plot_one_mode(NFs, EVs, my_mode, savefig=False):
    """INPUTS:
    NFs, EVs: the normal frequencies and eigenvectors as computed prior
    my_mode: mode we want to plot (integer)
    printfig: default is false; if true, will save a png
    """

    N = len(NFs)  # the total # of masses is also the # of frequencies
    y_n = np.zeros(N + 1)  # N+2 positions; +2 to add the left and right walls
    ftsz = 12  # font size

    seq = np.argsort(NFs)  # sorting eigenfrequencies in ascending order
    mm = seq[my_mode - 1]  # select the correct mode in the list
    Amp = abs(EVs[:, mm]).max()  # amplitude of the mode, more or less
    T = 2 * np.pi / NFs[mm]  # period of that mode
    time = np.linspace(0.0, 0.5 * T, 7)  # our time array; half a period

    rest_positions = range(N + 1)  # for our x-axis

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.grid()

    ax.set_xlim(0, N + 1)
    ax.set_ylim(-1.1 * Amp, 1.1 * Amp)
    ax.set_xlabel("$n$, the mass number", fontsize=ftsz)
    ax.set_ylabel("$y_n$, deviation from rest position", fontsize=ftsz)
    ax.set_title("Mode number {0:d} for {1:d} masses".format(my_mode, N), fontsize=ftsz)

    for n in range(N):  # the mass index; we leave the ends attached
        y_n[n + 1] = EVs[n, mm]

    for tt in time:
        ax.plot(
            rest_positions,
            y_n * np.cos(NFs[mm] * tt),
            "o-",
            label="$t = {0:.1f}T_{1:d}$".format(tt / T, my_mode),
        )
    # plt.legend()
    plt.tight_layout()

    if savefig:
        plt.savefig("{0:02d}masses_mode{1:02d}.png".format(N, my_mode))
        # png is easier to include in the written notes; pdf is best

    return


N = 20  # number of masses
m0 = 1.0  # [kg] the value of the masses
k0 = 1.0  # [N/m] the value of the stiffnesses

m = m0 * np.ones(N)  # a 1D numpy array of N masses filled with m0's
k = k0 * np.ones(N + 1)

print(k)

M = mass_matrix(N, m)  # the mass matrix
iM = la.inv(M)  # inverse of M
K = coupling_matrix(N, k)  # The matrix that couples oscillators

iMK = np.matmul(iM, K)  # the matrix, we seek the eigenvectors of

iMK[-1, -1] = 1

# Computing the eigenvectors
eigvals, eigvecs = la.eig(iMK)
eigfreqs = np.sqrt(eigvals)
seq = np.argsort(eigfreqs)  # sequence that sorts eigfreqs in ascending order

print("The sorted eigenfrequencies are")
print(eigfreqs[seq])

# We plot modes; in this example, every other mode starting from fundamental,
# stopping in the middle
for mode in range(1, N + 1, 1):
    plot_one_mode(eigfreqs, eigvecs, mode, savefig=True)

# plt.show()
