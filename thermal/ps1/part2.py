# ------------------------ #
# Standard Library Imports #
# ------------------------ #

import numpy as np                # Fundamental package for numerical computations
import matplotlib.pyplot as plt   # Plotting library
import heapq                      # Priority queue implementation

# --------------------------------- #
# Initialization of physical system #
# --------------------------------- #

NA = 100; NB = NA; N = NA + NB   # number of particles
density = 1.0                    # number density
L = float(N / density)           # box length
E = 1000.0                       # total energy
mA, mB = 1.0, 2.0                # masses of species A and B
kB = 1.0                         # Boltzmann constant in simulation units

np.random.seed(252)
# np.random.seed()

# positions: uniform spacing inside (0, L) plus jitter
jitter = 1/3; x = (L / N) * ( np.arange(0, N) + 0.5 + jitter * (np.random.rand(N) - 0.5) )
assert (np.all(x >= 0) and np.all(x <= L)), "Some positions are out of bounds!"
assert (np.all(np.diff(x) > 0)), "Some particles in incorrect order!"

# masses: alternating betwee A-B-A-B-...
m = np.empty(N); m[::2] = mA; m[1::2] = mB; Mtot = np.sum(m)

# velocities: use bimodal, but consider other initial distributions to check whether final distribution depends on it
c = 2.0; w = 1.0
v = np.concatenate([np.random.normal(loc=c, scale=1, size=NA), np.random.normal(loc=-c, scale=w, size=NB)])
np.random.shuffle(v)

# implement constraints: zero total momentum and desired total energy
v -= np.sum(m * v) / Mtot
v *= np.sqrt( E / (0.5 * np.sum(m * np.power(v,2))) )

# internal clock of each particle for lazy position updates
T = 0.0; t_last = np.full(N, T)

# ------------------------------------------------------------ #
# Useful functions and initialization of bookkeeping variables #
# ------------------------------------------------------------ #
ev_type = {"pair":0, "wall":1}; LEFT, RIGHT = 0, 1

####### free evolution 
def pos_at(i, T, t_last=t_last):
    # Update position of particle i at global time T
    return x[i] + v[i] * (T - t_last[i])

####### collisions between particles
# time to next collision between particles i and i+1
def next_pair_time(i, T):
    gap = pos_at(i+1, T) - pos_at(i, T); relv = v[i] - v[i+1]
    return T + gap / relv if relv > 0 else np.inf

# send it to the heap
def push_pair(i, T):
    ti = next_pair_time(i, T)
    pair_time[i] = ti; pair_gen[i] += 1
    heapq.heappush(heap, (ti, ev_type["pair"], i, pair_gen[i]))

# transform velocities upon collision between particles i and i+1
def collide_pair(i, T, t_last=t_last, x=x, v=v, m=m):
    # update positions to time T, where the collision occurs, in-place
    xi = pos_at(i, T); xj = pos_at(i+1, T)
    x[i] = xi; x[i+1] = xj
    assert np.allclose(xj - xi, 0.0), "Collision detected but particles not at same position!"
    # update internal clocks to time T, in-place
    t_last[i] = T; t_last[i+1] = T
    # update velocities according to elastic collision rules, in-place
    mi, mj = m[i], m[i+1]; vi, vj = v[i], v[i+1]
    v[i]   = vi + (2 * mj / (mi + mj)) * (vj - vi)
    v[i+1] = vj + (2 * mi / (mi + mj)) * (vi - vj)
    return None

####### collisions with walls
# time to next collision between particle 0 and left wall, or particle N-1 and right wall
def next_wall_time(side, T):
    i = N-1 if side else 0
    xi = pos_at(i, T); vi = v[i]
    if side:
        return T + (L - xi)/vi if vi > 0 else np.inf
    else:
        return T + xi/(-vi) if vi < 0 else np.inf

# send it to the heap
def push_wall(side, T):
    ti = next_wall_time(side, T)
    wall_time[side] = ti; wall_gen[side] += 1
    heapq.heappush(heap, (ti, ev_type["wall"], side, wall_gen[side]))

# transform velocities upon collision with wall
def reflect_wall(side, T, t_last=t_last, x=x, v=v):
    i = N-1 if side else 0
    xi = pos_at(i, T); x[i] = xi; t_last[i] = T
    vi_before = v[i]; v[i] = -v[i]  # elastic reflection
    return i, vi_before

####### create the event heap and initialize it
pair_time = np.full(N-1, np.inf); pair_gen  = np.zeros(N-1, dtype=np.int64)
wall_time = np.full(2, np.inf); wall_gen  = np.zeros(2, dtype=np.int64)
heap = []  # min-heap of (time, kind, index, generation)

push_wall(RIGHT, T); push_wall(LEFT, T)
for i in range(N-1):
    push_pair(i, T)

# ----------------------------- #
# Simulation parameters and run #
# ----------------------------- #

num_events = 2_000_000   # total number of events (pair collisions + wall reflections)
burn_in = 1_200_000      # events discarded for equilibration
verbosity = 250_000      # print status every verbosity events
sample_every = 500       # sample velocities every k events after burn-in

# initialize variables for sampling and pressure accumulation
samples_A, samples_B = [], []
impulse_left = 0.0; impulse_right = 0.0
T_burn = None; processed = 0

# # Plot a histogram of the initial velocity distribution
# plt.hist(v, bins=30, density=True)
# plt.title('Initial Velocity Distribution')
# plt.xlabel('Velocity')
# plt.ylabel('Density')
# plt.tight_layout()
# plt.grid()
# plt.savefig('temp_images/initial_velocity_distribution_random3.png')
# # plt.show()
# exit()

while processed < num_events:
    ti, kind, idx_evt, gen = heapq.heappop(heap)

    ########## check pathological cases
    if not np.isfinite(ti):
        break  # no more events to process
    if (kind == ev_type["pair"] and (gen != pair_gen[idx_evt] or ti != pair_time[idx_evt])) or \
       (kind == ev_type["wall"] and (gen != wall_gen[idx_evt] or ti != wall_time[idx_evt])):
        # heaps don’t support “decrease‑key” or "update-key" easily in Python, skip outdated entries
        continue

    T = ti
    if processed == burn_in:
        T_burn = T

    if kind == ev_type["pair"]:
        collide_pair(idx_evt, T)
        # local rescheduling
        if idx_evt - 1 >= 0: push_pair(idx_evt - 1, T)
        push_pair(idx_evt, T)
        if idx_evt + 1 <= N - 2: push_pair(idx_evt + 1, T)
        if idx_evt == 0: push_wall(LEFT, T)
        if idx_evt == N - 2: push_wall(RIGHT, T)
    else:
        i, vi_before = reflect_wall(idx_evt, T)
        if processed >= burn_in:
            # accumulate impulse for pressure calculation
            if idx_evt == 0:
                impulse_left += 2 * abs(vi_before) * m[i]
            else:
                impulse_right += 2 * abs(vi_before) * m[i]
        if idx_evt == 0:
            push_pair(0, T); push_wall(LEFT, T)
        else:
            push_pair(N - 2, T); push_wall(RIGHT, T)

    processed += 1

    if processed >= burn_in and (processed - burn_in) % sample_every == 0:
        samples_A.extend(v[m == mA])
        samples_B.extend(v[m == mB])

    if processed % verbosity == 0:
        print(f"Processed {processed:,} events...")

# Determine the pressure from the gas
time_total = T - T_burn
area = 1.0  # set area to 1 because this is a 1D simulation
pressure_left = impulse_left / (area * time_total)
pressure_right = impulse_right / (area * time_total)
pressure_avg = 0.5 * (pressure_left + pressure_right)

# Print pressure results
print(f"\nPressure on left wall: {pressure_left:.4e}")
print(f"Pressure on right wall: {pressure_right:.4e}")
print(f"Average pressure: {pressure_avg:.4e}\n")

# Save samples_A and samples_B to files
np.save(f"temp_images/samples_A.npy", np.array(samples_A))
np.save(f"temp_images/samples_B.npy", np.array(samples_B))
