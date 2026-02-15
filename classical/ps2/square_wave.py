import numpy as np
from matplotlib import pyplot as plt

x = np.linspace(-5 * np.pi / 2, 5 * np.pi / 2, 1000)
indices = range(1, 128, 2)

# Compute the Fourier series approximation of the square wave function
a_n = [4/(n * np.pi) * (-1)**((n-1)/2) for n in indices]
cosines = [np.cos(n * x) for n in indices]

square_wave_fourier = np.dot(a_n, cosines)

# Plot the square wave and its Fourier series approximation
plt.plot(x, square_wave_fourier, label='Fourier Series Approximation', color='blue')

# Labels
plt.title('Fourier Series Approximation of a Square Wave')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid()
plt.legend()
plt.show()