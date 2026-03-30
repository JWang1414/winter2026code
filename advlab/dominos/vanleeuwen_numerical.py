# This code is originally written in C++, from https://www.lorentz.leidenuniv.nl/lunchcalc/dominoes/
# The code is based on the paper van Leeuwen, J. M. J. The domino effect. American Journal of Physics 78(7) (2010), 721–727.

import math

import matplotlib.pyplot as plt
import numpy as np


def coefficients(n, x):  # based on equation (25)
    global A
    global B
    global C
    y = math.sqrt(1 - x * x)
    r = 1.0
    w = 1.0
    v = 0.0  # initial values w=psi', v=psi"
    A = 1.0
    B = 0.0
    C = 0.5 * (x - d * y)  # initial values of coefficients
    for j in range(1, n, 1):
        X = (sep + d) * y - d
        Y = math.sqrt(1 - X * X)
        xn = Y * x + X * y
        yn = -X * x + Y * y  # next angle
        T = 0.5 * (xn - d * yn)  # new torque
        ici = 1 / Y
        ti = X * ici
        zi = 1 - (sep + d) * x * ici
        ai = zi - mu * d * ici
        bi = 1 + mu * ti
        if ai < 0:
            break
        else:
            r *= ai / bi  # if ai<0 end of recursion
        if zi < 0:
            wn = 0
            v = 0
        else:
            wn = w * zi
            v *= zi
            v += ti * (wn - w) * (wn - w) - (sep + d) * y * w * w * ici
        A += r * wn
        B += r * v
        C += r * T  # formation of coefficients
        x = xn
        y = yn
        w = wn  # input for next step in loop


def derivative(n, i, oma):  # based on equation (24)
    coefficients(
        n, math.sin((i + 0.5) * dth)
    )  # coefficients A,B,C at theta=(i+0.5)*dth
    return (C / oma - B * oma) / (
        A + 0.5 * dth * (C / (oma**2) + B)
    )  # 3rd order scheme


def time(n, oma):  # simpson's rule, based on equation (15)
    global duration
    summ = 1 / oma
    s = 1
    for i in range(Ni + 1):
        oma += dth * derivative(n, i, oma)  # update angular velocity
        summ += (s + 3 if (i < Ni) else 1) / oma  # Simpson weights
        s = -s
    duration = summ * dth / 3
    return oma  # the angular velocity just before the next collision


def collision(n, oma):  # the next angular velocity, equation(30)
    coefficients(n + 1, 0)  # coefficients at theta=0, computes A(0)
    return oma * (A - 1) / A  # new starting value


# constants:
mu = 0.25  # friction
h = 48e-3  # height of  the domino
d = 7.5e-3 / h  # thickness divded by height of the dominoes
tau = math.sqrt(
    (0.048**2 + 0.0075**2) / 3 / (9.8 * 0.048)
)  # the time scale, equation (12)
g = 9.8  # gravitational acceleration
# parameter:
Nt = 40  # number of separations s/h
# global variables:
Ni = 0
duration = 0  # time
sepp = np.linspace(0.01, 0.6, Nt)  # ratio between spacing and height
dth = 0  # theta step size
A = 0
B = 0
C = 0  # coefficients in equation (25)
separation = []
asymp_vel = []
for i in sepp:
    Ni = 100  # number of intervals for integration
    Nc = 100  # number of collisions such that it is enough for convergence
    oma = 1.0  # initial angular velocity (not relevant for asymptotic behaviours)
    sep = i  # the current separation as a ratio of spacing and height
    thetac = math.asin(sep)  # critical angle (where collision occurs)
    dth = thetac / Ni  # theta step size
    delta_t = np.array([])  # stores the time interval between collisions
    for n in range(1, Nc, 1):
        omp = time(n, oma)  # updated angular velocity
        delta_t = np.append(delta_t, duration)  # note the time scale from equation (12)
        oma = collision(n, omp)  # intial angular velocity of next domino
    # plt.figure(i)
    # plt.plot(delta_t*tau)
    # plt.xlabel("collision number")
    # plt.ylabel("Time interval between collisions (s)")
    # plt.show()
    # plt.figure(i+Nt)
    # plt.plot([np.sum(delta_t[:n]*tau) for n in range(0, Nc-1,1)])
    # plt.xlabel("collision number")
    # plt.ylabel("Total time (s)")
    v_asymp = h * (sep + d) / (duration * tau) / math.sqrt(9.8 * h)
    asymp_vel.append(v_asymp)
    separation.append(sep)
    # print(sep, v_asymp)

print("Separation, Asymptotic Velocity")
for sep, v_asymp in zip(separation, asymp_vel):
    print(f"{sep:.3f}, {v_asymp:.3f}")

# plt.figure(10000)
# plt.plot(separation, asymp_vel)
# plt.title("Numerical results for Asymptotic Velocity vs Spacing to Height Ratio")
# plt.ylabel("Non-dimensionalized asymptotic velocity")
# plt.xlabel("Ratio of spacing to height")
# plt.show()
