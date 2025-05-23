from share_pulse_sim_funcs import *
from qutip import Options

wc_A = 4.069814 * (10**9) * 2 * np.pi  # cavity A frequency
wc_B = 6.096062 * (10**9) * 2 * np.pi  # cavity A frequency
wa =  5.325 * (10**9) * 2 * np.pi  # atom frequency
dt_A = np.abs(wc_A - wa) / (2 * np.pi)
dt_B = np.abs(wc_B - wa) / (2 * np.pi)
chi_A = 0.00215 * (10**9) * 2 * np.pi
chi_B = 0.00544 * (10**9) * 2 * np.pi
g_A = np.sqrt(chi_A * dt_A) * 2 * np.pi  # coupling strength w/ cavity A
g_B = np.sqrt(chi_B * dt_B) * 2 * np.pi  # coupling strength w/ cavity B

gamma = 333333.333        # atom dissipation rate
kappa_A = 10000       # cavity A dissipation rate
kappa_B = 10000       # cavity B dissipation rate

temp_q = 0.01        # avg number of thermal bath excitation for qubit
temp_A = 0.04        # avg number of thermal bath excitation for cavity A
temp_B = 0.05        # avg number of thermal bath excitation for cavity B

# Reward functions
def qubit_excited(final_expect, final_dm):
    return final_expect[0][-1]

def cavity_A_1Fock(final_expect, final_dm):
    return np.power(np.abs(final_dm[1]), 2)

# =================
# ==== OPTIONS ====
# =================

us = 0.000001
time_start = 0.0 * us
time_stop = 0.1 * us

cavity_dims = 8
num_cavities = 1
state_sizes = [2, cavity_dims]
state_vals = [0.0, 0.0]
sim_options = {
    "store_final_state": True,
}

element_frequencies = [wa, wc_A]
drive_frequencies = [wa, wc_A]
drive_element_indices = [0, 1]
reward_function = cavity_A_1Fock
verbose = True
plot_trial_pulses = False


initial_state = tensor((basis(state_sizes[0], 0) * np.sqrt(1 - state_vals[0])) +
                    (basis(state_sizes[0], 1) * np.sqrt(state_vals[0])),
                       (basis(state_sizes[1], 0) * np.sqrt(1 - state_vals[1])) +
                       (basis(state_sizes[1], 1) * np.sqrt(state_vals[1])))

sm, a_A, a_B, sx, sz = reg_ops(num_cavities + 1, cavity_dims)

drive_operators = [sm.dag(), sm, a_A.dag(), a_A]
element_ops = [sz, a_A.dag() * a_A]
H_0 = (chi_A * a_A.dag() * a_A * sz / 2)
evaluation_operators = [sm.dag() * sm, a_A.dag() * a_A]

args = {
    "base_hamiltonian": H_0,
    "options": sim_options,
    "time_start": time_start,
    "time_stop": time_stop,
    "initial_state": initial_state,
    "element_frequencies": element_frequencies,
    "drive_frequencies": drive_frequencies,
    "drive_element_indices": drive_element_indices,
    "drive_operators": drive_operators,
    "evaluation_operators": evaluation_operators,
    "reward_function": reward_function,
    "verbose": verbose,
    "plot_trial_pulses": plot_trial_pulses,
}


# =========================
# ==== Drive functions ====
# =========================

test_amps = np.linspace(0, 100000000, 101)

test_rewards = []

for A in test_amps:

    qubit_drive_amplitude = 0
    qubit_drive_mu = 0.05 * us
    qubit_drive_sigma = 0.02 * us

    cavity_drive_amplitude = A
    cavity_drive_mu = 0.05 * us
    cavity_drive_sigma = 0.025 * us

    drive_funcs = [gauss(qubit_drive_amplitude, qubit_drive_mu, qubit_drive_sigma, 0), zero, gauss(cavity_drive_amplitude, cavity_drive_mu, cavity_drive_sigma, 0), zero] # This array requires 2 functions for each drive, one real, and one imaginary

    test_return = sim_interp_cost_eval(drive_funcs, args)
    test_rewards.append(test_return)

plt.plot(test_amps, test_rewards)
plt.show()