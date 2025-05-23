import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as inte
import time
import os
from math import floor
from qutip import mesolve, QobjEvo
from qutip import tensor, destroy, qeye, sigmax, sigmaz, basis
import numbers

def colors(ch):
    """
    Just a list of colors, useful for making a graph and wanting some colors to be the same and others to be different

    :param ch: Index of color to select, colors inputted arbitrarily
    :return:
    """
    if ch == 0:
            return 'tab:blue'
    elif ch == 1:
            return 'tab:orange'
    elif ch == 2:
            return 'tab:green'
    elif ch == 3:
            return 'tab:red'
    elif ch == 4:
            return 'tab:purple'
    elif ch == 5:
            return 'tab:brown'
    elif ch == 6:
            return 'tab:pink'
    elif ch == 7:
            return 'tab:gray'
    elif ch == 8:
            return 'tab:olive'
    elif ch == 9:
            return 'tab:cyan'

    return 'black'

def iq_to_amp(i_vals, q_vals, conj=False):
    if np.shape(i_vals) != np.shape(q_vals):
        raise ("Number of I and Q value functions must be the same")
    if isinstance(i_vals, (np.ndarray, list, int, float, tuple, complex)):
        if conj:
            return i_vals - (1j * q_vals)
        else:
            return i_vals + (1j * q_vals)
    else:
        if conj:
            def amp_helper(t):
                return i_vals(t) - (1j * q_vals(t))
        else:
            def amp_helper(t):
                return i_vals(t) + (1j * q_vals(t))
        return amp_helper


def drive_osc_camp(amp_func, phase_func, w0, wd, conj=False):
    if conj:
        def drive_gen_helper(*t):
            t = t[0]
            return amp_func(t) * np.exp(1j * (((wd - w0) * t) + phase_func(t)))
    else:
        def drive_gen_helper(*t):
            t = t[0]
            return amp_func(t) * np.exp(1j * (((w0 - wd) * t) + phase_func(t)))
    return drive_gen_helper

def zero(t):
    return t - t

def ones(x):
    return (x + 1) - x

def gauss(A, mu, sigma, C=0):
    def ret_gauss(x):
        def x_map(y):
            if ((y >= (mu - (2 * sigma))) and (y <= (mu + (2 * sigma)))):
                return A * np.exp(-np.power(y - mu, 2) / (2 * np.power(sigma, 2))) + C
            return 0

        if(isinstance(x, numbers.Number)):
            return x_map(x)

        return list(map(x_map, x))

    return ret_gauss

def func_add(func1, func2):
    def added_funcs(x):
        return np.array(func1(x)) + np.array(func2(x))

    return added_funcs


def reg_ops(n_elem, N):
    r_ops = []
    if n_elem == 1:
        r_ops.append(tensor(destroy(2)))
        r_ops.append(tensor(qeye(2)))
        r_ops.append(tensor(qeye(2)))
        r_ops.append(tensor(sigmax()))
        r_ops.append(tensor(sigmaz()))
    if n_elem == 2:
        r_ops.append(tensor(destroy(2), qeye(N)))
        r_ops.append(tensor(qeye(2), destroy(N)))
        r_ops.append(tensor(qeye(2), qeye(N)))
        r_ops.append(tensor(sigmax(), qeye(N)))
        r_ops.append(tensor(sigmaz(), qeye(N)))
    elif n_elem == 3:
        r_ops.append(tensor(destroy(2), qeye(N), qeye(N)))
        r_ops.append(tensor(qeye(2), destroy(N), qeye(N)))
        r_ops.append(tensor(qeye(2), qeye(N), destroy(N)))
        r_ops.append(tensor(sigmax(), qeye(N), qeye(N)))
        r_ops.append(tensor(sigmaz(), qeye(N), qeye(N)))
    return r_ops


def sim_interp_cost_eval(drive_funcs, args):
    if len(args) == 1:
        args = args[0]
    H_0 = args["base_hamiltonian"]
    options = args["options"]
    time_start = args["time_start"]
    time_stop = args["time_stop"]
    init_state = args["initial_state"]
    freqs = args["element_frequencies"]
    drive_freqs = args["drive_frequencies"]
    drive_elem_nums = args["drive_element_indices"]
    drive_ops = args["drive_operators"]
    eval_ops = args["evaluation_operators"]
    reward_function = args["reward_function"]
    verbose = args["verbose"]
    plot_trial_pulses = args["plot_trial_pulses"]

    t_arr = np.linspace(time_start, time_stop, 1001)

    num_drives = len(drive_elem_nums)

    # Plot out the amplitudes of the drives
    if plot_trial_pulses:
        for i in range(num_drives):
            plt.plot(t_arr, drive_funcs[(2 * i)](t_arr), label=f'Drive Re({i})', color=colors(i), linestyle="solid")
            plt.plot(t_arr, drive_funcs[(2 * i) + 1](t_arr), label=f'Drive Im({i})', color=colors(i),
                     linestyle="dotted")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.title("Drive pulses")
        plt.legend()
        save_dir = r'C:\_Data\images\20250509'
        img_name = time.strftime('%Y%m%d-%H%M%S.png')
        plt.savefig(os.path.join(save_dir, img_name))
        print(f'Saved pulse plot to {os.path.join(save_dir, img_name)}')
        plt.close()

    drive_camps = []
    for i in range(0, 2 * num_drives, 2):
        drive_camps.append(iq_to_amp(drive_funcs[i], drive_funcs[i + 1]))
        drive_camps.append(iq_to_amp(drive_funcs[i], drive_funcs[i + 1], conj=True))

    drive_osc_camps = []

    for i in range(num_drives * 2):
        drive_osc_camps = [drive_osc_camp(drive_camps[i], zero, freqs[drive_elem_nums[floor(i / 2)]], drive_freqs[floor(i / 2)]) for i in
                       range(num_drives * 2)]

    # if plot_trial_pulses:
    #     for i in range(num_drives):
    #         plt.plot(t_arr, np.abs(drive_osc_camps[(2 * i)](t_arr)), label=f'Drive Re({i})', color=colors(i), linestyle="solid")
    #         plt.plot(t_arr, np.abs(drive_osc_camps[(2 * i) + 1](t_arr)), label=f'Drive Im({i})', color=colors(i),
    #                  linestyle="dotted")
    #         print(drive_funcs[(2 * i)](t_arr)[0])
    #         print(drive_funcs[(2 * i) + 1](t_arr)[0])
    #     plt.xlabel("Time")
    #     plt.ylabel("Amplitude")
    #     plt.title("Drive pulses")
    #     plt.legend()
    #     save_dir = r'C:\_Data\images\20241205'
    #     img_name = time.strftime('%Y%m%d-%H%M%S.png')
    #     plt.savefig(os.path.join(save_dir, img_name))
    #     plt.close()

    H_t = [[drive_ops[i], drive_osc_camps[i]] for i in range(len(drive_osc_camps))]

    H = [H_0] + H_t
    H = QobjEvo(H, tlist=t_arr)

    print("init_state:", init_state)

    res = mesolve(H, init_state, t_arr, c_ops=[], e_ops=eval_ops, options=options)

    final_expect = res.expect

    final_dm = res.final_state

    if verbose:
        print("final_expect:", final_expect)
        print(f'final_dm: {final_dm}')

    reward = np.abs(reward_function(final_expect, final_dm))

    if verbose:
        print(f'Reward: {reward}')

    return reward