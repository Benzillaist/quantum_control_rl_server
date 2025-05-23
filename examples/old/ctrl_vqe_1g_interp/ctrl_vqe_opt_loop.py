import numpy as np
from ctrl_vqe_pi_pulse_client import pulse_client
from ctrl_vqe_pi_pulse_training_server import start_training_server
import subprocess
import time
import copy
from utils.opt_utils import *

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

tlist = np.linspace(0, 0.000003, 501)

# Cost function
def cost_q_e(final_expect, final_dm):
    print(final_expect[0])
    return(final_expect[0])

# ========== OPTIONS ========== #
max_segs = 10
time_start = 0.0000000
time_stop = 0.000001
init_amp = 10000
n_steps = 501

qamp_u_bound = 1000000000
qamp_l_bound = -1000000000
camp_u_bound = 1000000000
camp_l_bound = -1000000000

num_drives = 3
num_elems = 0
num_cavities = 2
cavity_dims = 6
state_sizes = [2, cavity_dims, cavity_dims]
state_vals = [temp_q, temp_A, temp_B]
init_freqs = [wa] # [wa, wc_A, wc_B]
sim_options = Options()
element_freqs = [wa] # [wa, wc_A, wc_B]
output_cost_func = cost_stoAB_g11
verbose = True

epochs = 100
train_batch_size = 20
freq_scale = 0.0015

opt_options = {"xatol": 1, "ftol": 4e-2, "xtol": 1, "finite_diff_rel_step": 100, "stepmx": 1000, "scale":np.array([1.0, 1.0, 1000.0])}
# ========== OPTIONS ========== #

t_step = (time_stop - time_start) / n_steps

qscale = []
cscale = []
for i in range(2):
    qscale.append((qamp_u_bound - qamp_l_bound) / 2)
    # cbounds.append((camp_l_bound, camp_u_bound))
    # cbounds.append((camp_l_bound, camp_u_bound))

sm, a_A, a_B, sx, sz = reg_ops(num_cavities + 1, cavity_dims)
drive_freqs = np.array(init_freqs)

gammas = [gamma, kappa_A, kappa_B]
temps = [temp_q, temp_A, temp_B]
c_ops = gen_c_ops("q", [sm, a_A, a_B, sx, sz], gammas, temps)

# Operators used in Hamiltonian
drive_ops = [sm.dag(), sm, a_A.dag(), a_A, a_B.dag(), a_B]
element_ops = [] # [sz, a_A.dag() * a_A, a_B.dag() * a_B]
H_0 = (chi_A * a_A.dag() * a_A * sz / 2) + (chi_B * a_B.dag() * a_B * sz / 2)
eval_ops = [sm.dag() * sm, a_A.dag() * a_A, a_B.dag() * a_B, tensor(destroy(2) * destroy(2).dag(), destroy(cavity_dims).dag() * destroy(cavity_dims), destroy(cavity_dims).dag() * destroy(cavity_dims))]

t_segs, amp_segs = setup_segs(2 * num_drives, time_start, time_stop, init_amp)

# Setup initial state
# init_state = build_psi(state_sizes, state_vals)
init_state = tensor((basis(state_sizes[0], 0) * np.sqrt(1 - state_vals[0])) + (basis(state_sizes[0], 1) * np.sqrt(state_vals[0])), (basis(state_sizes[1], 0) * np.sqrt(1 - state_vals[1])) + (basis(state_sizes[1], 1) * np.sqrt(state_vals[1])), (basis(state_sizes[2], 0) * np.sqrt(1 - state_vals[2])) + (basis(state_sizes[2], 1) * np.sqrt(state_vals[2])))

# Create blank history arrays for storing optimal / past values
time_hist = []
amp_hist = []
freq_hist = []
cost_hist = []
meth_hist = []

# Run vqe, etc
for i in range(max_segs):

    temp_amp_scale = copy.deepcopy(qscale)
    temp_amp_scale = temp_amp_scale + cbounds
    # temp_bounds.append(0.0015 * drive_w_arr[0])
    # temp_bounds.append((1 * drive_w_arr[1], 1 * drive_w_arr[1]))
    # temp_bounds.append((1 * drive_w_arr[2], 1 * drive_w_arr[2]))

    init_amps = amp_segs.flatten()
    init_freqs = drive_freqs
    client_args = (num_drives, num_elems, drive_ops, t_segs, t_step, H_0, init_state, c_ops, eval_ops, sim_options, element_ops, element_freqs, output_cost_func, verbose)
    server_args = (epochs, train_batch_size, time_segs, init_amps, init_freqs, temp_amp_scale, list(freq_scale * np.array(drive_freqs)))

    # Save args for rl client
    np.savetxt("temp_files/client_args.txt", client_args)

    # Save args for rl server
    np.savetxt("temp_files/server_args.txt", server_args)

    os.system('cmd /c python ./run_rl_scripts.py')


    if res_PowNM.fun < res_NM.fun:
        # updates amplitudes and frequencies with optimized values and reshape
        if num_elems > 0:
            amp_segs = res_PowNM.x[:-num_elems]
            drive_freqs = res_PowNM.x[-num_elems:]
        else:
            amp_segs = res_PowNM.x
            drive_freqs = []
        # save values to history arrays
        cost_hist.append(res_PowNM.fun)
        meth_hist.append("PowNM")
    else:
        # updates amplitudes and frequencies with optimized values and reshape
        if num_elems > 0:
            amp_segs = res_NM.x[:-num_elems]
            drive_freqs = res_NM.x[-num_elems:]
        else:
            amp_segs = res_NM.x
            drive_freqs = []
        # save values to history arrays
        cost_hist.append(res_NM.fun)
        meth_hist.append("NM")

    # updates amplitudes and frequencies with optimized values and reshape
    amp_segs = np.reshape(amp_segs, (num_drives * 2, int(len(amp_segs) / (2 * num_drives))))

    # save values to history arrays
    time_hist.append(t_segs)
    amp_hist.append(amp_segs)
    freq_hist.append(drive_freqs)

    for i in range(2):
        qbounds.append((qamp_l_bound, qamp_u_bound))
        cbounds.append((camp_l_bound, camp_u_bound))
        cbounds.append((camp_l_bound, camp_u_bound))

    # split segments and return to start of loop
    if (i < max_segs - 1):
        print(t_segs)
        print(amp_segs)
        t_segs, amp_segs = split_segs(t_segs, amp_segs)