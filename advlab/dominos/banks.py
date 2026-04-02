import math as m

import numpy as np
import scipy.integrate as integrate

# Python implementation of Banks model for the domino effect
# This model only considers the effect of one domino on the next and does not
# consider the whole chain of dominoes. As well the collision is elastic.

# ratio_s = 2/6 # s/h ratio of distance between dominoes to the height of
# the domino
h = 4.8  # cm height of domino
d = 0.75  # cm width/depth of domino
# sep = ratio_s * h # cm separation between dominoes
g = 9.81  # m/s^2 gravity

ratio_d = d / h  # d/h aspect ratio of dominoes


def omega_max(sep, g, h):
    # maximum value for angular velocity. This would be the angular velocity
    # when the physical pendulum is at the bottom of a swing
    sintheta = sep / h
    costheta = m.sqrt(h**2 - sep**2) / h
    const = (3 * g) / h
    return (1 / sintheta) * m.sqrt(const * (2 - (1 + costheta) * costheta**2))


def time(max_omega, theta_c, g, h):
    # compute the time it takes for a domino to go from vertical to collision
    # angle theta_c
    k = m.sqrt(6 * g / h) / max_omega
    phi_c = (m.pi - theta_c) / 2
    comp_ell_int = ell_int(k, 0, m.pi / 2)  # K
    incomp_ell_int = ell_int(k, 0, phi_c)  # F
    return (2 / max_omega) * (comp_ell_int[0] - incomp_ell_int[0])


def ell_func(phi, k):
    """Function we want to integrate"""
    # test = k ** 2 * m.sin(phi) ** 2
    value = 1 / m.sqrt(1 - k**2 * m.sin(phi) ** 2)
    return value


def ell_int(k, a, b):
    """Elliptical integral integration
    phi is the variable we are integrating
    """
    return integrate.quad(ell_func, a, b, args=k)


num_sep = 80
ratio_s_list = np.linspace(0, 0.6, num_sep)
values = []

for ratio_s in ratio_s_list:
    sep = ratio_s * h
    theta_c = m.asin(ratio_s)  # ratio_s is s/h
    max_angular_velocity = omega_max(sep, g, h)
    dimensionless_vel = (sep / time(max_angular_velocity, theta_c, g, h)) / m.sqrt(
        g * h
    )
    values.append((ratio_s, dimensionless_vel))

print("Separation, Asymptotic Velocity")
for ratio_s, dimensionless_vel in values:
    print(f"{ratio_s:.3f}, {dimensionless_vel:.3f}")
