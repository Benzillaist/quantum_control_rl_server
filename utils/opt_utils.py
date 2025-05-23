# Helper functions
import numpy as np
import matplotlib.pyplot as plt
from qutip import tensor, destroy, qeye, sigmax, sigmaz, basis, mesolve
from IPython.display import clear_output
import scipy.interpolate as inte
from math import floor
from qutip import mesolve, QobjEvo
import utils.pulse_configs as pc
from utils.hamiltonians import ArbHamiltonianEval
from qick import QickConfig
import Pyro4
import copy
from scipy.optimize import curve_fit
import time
import os

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


def gen_c_ops(arg, ops, gammas, temps):
    sm, a_A, a_B, sx, sz = ops
    c_ops = []

    if "q" in arg:
        # qubit relaxation
        rate = gammas[0] * (1 + temps[0])
        if rate > 0.0:
            c_ops.append(np.sqrt(rate) * sm)

        # qubit excitation, if temperature > 0
        rate = gammas[0] * temps[0]
        if rate > 0.0:
            c_ops.append(np.sqrt(rate) * sm.dag())

    if "A" in arg:
        # cavity A relaxation
        rate = gammas[1] * (1 + temps[1])
        if rate > 0.0:
            c_ops.append(np.sqrt(rate) * a_A)

        # cavity A excitation, if temperature > 0
        rate = gammas[1] * temps[1]
        if rate > 0.0:
            c_ops.append(np.sqrt(rate) * a_A.dag())

    if "B" in arg:
        # cavity B relaxation
        rate = gammas[2] * (1 + temps[2])
        if rate > 0.0:
            c_ops.append(np.sqrt(rate) * a_B)

        # cavity B excitation, if temperature > 0
        rate = gammas[2] * temps[2]
        if rate > 0.0:
            c_ops.append(np.sqrt(rate) * a_B.dag())

    return c_ops


def gauss_gen(A, mu, sigma, C):
    def gauss_helper(x):
        return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + C

    return gauss_helper


def sin_gen(phase_func, w):
    def sin_helper(x):
        return np.sin(phase_func(x) + (w * x))

    return sin_helper


def rect_seg(amp, ti, tf):
    def pulse_helper(t):
        if not isinstance(t, np.ndarray):
            t = np.array(t)
        tc = np.zeros(np.shape(t))
        tc[(t >= ti) & (t <= tf)] = amp
        return tc

    return pulse_helper


def interp_seg(amps, times):

    spline = inte.CubicSpline(times, amps, bc_type='clamped')
    ti, tf = times
    def pulse_helper(t):
        if not isinstance(t, np.ndarray):
            t = np.array(t)
        tc = np.zeros(np.shape(t))
        tc[(t >= ti) & (t <= tf)] = spline(t[(t >= ti) & (t <= tf)])
        return tc

    return pulse_helper



def gauss_seg(amp, sig, ti, tf, C=0):
    def pulse_helper(t):
        if not isinstance(t, np.ndarray):
            t = np.array(t)
        tc = np.zeros(np.shape(t))
        t_inc = t[(t >= ti) & (t <= tf)]
        # print(f'len(tc): {len(tc)}')
        # print(f'len(t_inc): {len(t_inc)}')
        tc[(t >= ti) & (t <= tf)] = amp * np.exp(-(t_inc - ((tf + ti) / 2)) ** 2 / (2 * sig ** 2)) + C
        return tc

    return pulse_helper


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


def func_sum(funcs):
    if len(funcs) == 0:
        return zero

    def sum_helper(t):
        ret_sum = funcs[0](t)
        for i, func in enumerate(funcs):
            if i != 0:
                ret_sum += func(t)
        return ret_sum

    return sum_helper


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


def iq_to_phase(i_func, q_func):
    def phase_helper(t):
        return np.arctan2(i_func(t), q_func(t))

    return phase_helper


def zero(t):
    return t - t


# setup initial config and segment
def setup_segs(n_drives, time_start, time_stop, init_amp):
    t_segs = np.empty((n_drives, 2))
    for i in range(n_drives):
        t_segs[i] = [time_start, time_stop]
    amp_segs = np.full((n_drives, 1), init_amp)
    return t_segs, amp_segs

def setup_interp_segs(n_drives, time_start, time_stop, init_amp):
    t_segs = np.empty((n_drives, 1))
    for i in range(n_drives):
        t_segs[i] = [(np.random.rand(1)[0] * (time_stop - time_start)) + time_start]
    amp_segs = np.full((n_drives, 1), init_amp)
    return interp_time_wrapper(t_segs, time_start, time_stop), amp_segs


# Splits up time and amplitudes of time segments after each optimization iteration
def split_segs(time_mat, amp_mat):
    """

    :param time_mat:
    :param amp_mat:
    :return:
    """
    time_mat_shape = np.shape(time_mat)
    random_arr = np.random.rand(time_mat_shape[0])
    extend_time_mat = np.empty((time_mat_shape[0], time_mat_shape[1] + 1))
    extend_amp_mat = np.empty((time_mat_shape[0], time_mat_shape[1] - 1))
    for i, time_arr in enumerate(time_mat):
        max_index = np.argmax(time_arr[1:] - time_arr[:-1])
        if max_index == 0:
            rand_time = time_arr[max_index] + (random_arr[i] * (time_arr[max_index + 1] - time_arr[max_index]))
            extend_time_mat[i] = np.insert(time_arr, max_index + 1, rand_time)
            a = np.insert(amp_mat[i], max_index + 1, (amp_mat[i][max_index] * (rand_time - time_arr[max_index]) + amp_mat[i][max_index + 1] * (time_arr[max_index + 1] - rand_time)) / (time_arr[max_index + 1] - time_arr[max_index]))
            extend_amp_mat[i] = a
        else:
            rand_time = time_arr[max_index] + (random_arr[i] * (time_arr[max_index + 1] - time_arr[max_index]))
            extend_time_mat[i] = np.insert(time_arr, max_index + 1, rand_time)
            a = np.insert(amp_mat[i], max_index + 1, (amp_mat[i][max_index] * (rand_time - time_arr[max_index]) + amp_mat[i][max_index + 1] * (time_arr[max_index + 1] - rand_time)) / (time_arr[max_index + 1] - time_arr[max_index]))
            extend_amp_mat[i] = a
    return extend_time_mat, extend_amp_mat

# Splits up time and amplitudes of time segments after each optimization iteration
def split_segs_flat(time_mat, amp_mat):
    """

    :param time_mat:
    :param amp_mat:
    :return:
    """
    time_mat_shape = np.shape(time_mat)
    random_arr = np.random.rand(time_mat_shape[0])
    extend_time_mat = np.empty((time_mat_shape[0], time_mat_shape[1] + 1))
    extend_amp_mat = np.empty((time_mat_shape[0], time_mat_shape[1] - 1))
    for i, time_arr in enumerate(time_mat):
        max_index = np.argmax(time_arr[1:] - time_arr[:-1])
        if max_index == 0:
            rand_time = time_arr[max_index] + (random_arr[i] * (time_arr[max_index + 1] - time_arr[max_index]))
            extend_time_mat[i] = np.insert(time_arr, max_index + 1, rand_time)
            a = np.insert(amp_mat[i], max_index + 1, (amp_mat[i][max_index] * (rand_time - time_arr[max_index]) + amp_mat[i][max_index + 1] * (time_arr[max_index + 1] - rand_time)) / (time_arr[max_index + 1] - time_arr[max_index]))
            extend_amp_mat[i] = a[1:-1]
        else:
            rand_time = time_arr[max_index] + (random_arr[i] * (time_arr[max_index + 1] - time_arr[max_index]))
            extend_time_mat[i] = np.insert(time_arr, max_index + 1, rand_time)
            a = np.insert(amp_mat[i], max_index + 1, (amp_mat[i][max_index] * (rand_time - time_arr[max_index]) + amp_mat[i][max_index + 1] * (time_arr[max_index + 1] - rand_time)) / (time_arr[max_index + 1] - time_arr[max_index]))
            extend_amp_mat[i] = a[1:-1]
    return extend_time_mat, extend_amp_mat

def split_segs_interp(time_mat, amp_mat, ti, tf):
    zero_row = np.full((np.shape(time_mat)[0], 1), 0)
    ti_row = np.full((np.shape(time_mat)[0], 1), ti)
    tf_row = np.full((np.shape(time_mat)[0], 1), tf)

    temp_time_mat = np.append(np.append(ti_row, time_mat, axis=1), tf_row, axis=1)
    temp_amp_mat = np.append(np.append(zero_row, amp_mat, axis=1), zero_row, axis=1)

    ret_time_mat, ret_amp_mat = split_segs(temp_time_mat, temp_amp_mat)

    return ret_time_mat[:, 1:-1], ret_amp_mat[:, 1:-1]

def interp_time_wrapper(time_mat, ti, tf):
    ti_row = np.full((np.shape(time_mat)[0], 1), ti)
    tf_row = np.full((np.shape(time_mat)[0], 1), tf)
    return np.append(np.append(ti_row, time_mat, axis=1), tf_row, axis=1)

def interp_amp_wrapper(amp_mat):
    zero_row = np.full((np.shape(amp_mat)[0], 1), 0)
    return np.append(np.append(zero_row, amp_mat, axis=1), zero_row, axis=1)


def amp_by_time(time, time_arr, amp_arr):
    """
    For a sparsely defined array of amplitudes, find the amplitude that will be played at any specific point in time

    :param time: Time to get the amplitude at
    :param time_arr: Times that the pulse starts/changes
    :param amp_arr: Amplitudes corresponding to times in time_arr
    :return:
    """
    if len(time_arr[(time - time_arr) >= 0]) == 0:
        return 0
    if len(time_arr) > len(amp_arr):
        time_arr = time_arr[:len(amp_arr)]
    return amp_arr[(time - time_arr) >= 0][-1]


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

def linestyles(ch):
    if ch == 0:
            return 'solid'
    elif ch == 1:
            return 'dotted'
    elif ch == 2:
            return 'dashed'
    elif ch == 3:
            return 'dashdot'
    elif ch == 4:
            return (0, (1, 10))
    elif ch == 5:
            return (0, (5, 5))
    elif ch == 6:
            return (0, (3, 5, 1, 5))
    elif ch == 7:
            return (0, (3, 10, 1, 10))
    elif ch == 8:
            return (0, (3, 5, 1, 5, 5))
    elif ch == 9:
            return (0, (3, 1, 1, 1, 1, 1))

    return 'solid'

def qick_spread_pulse_amps(soccfg, t_block, amp_func, ch):
    """
    Creates an array of values that the QiCK board can use in creating an arbitrary waveform

    :param soccfg: Soc config of the QiCK board
    :param t_block: Times that the pulse will be played/changed at  # TODO: Change to just include the start and stop times
    :param amp_func: Function that returns the amplitude of the pulse for an inputted time
    :param ch: DAC channel on which the pulse is played on
    :return:
    """
    gencfg = soccfg['gens'][ch]
    samps_per_clk = gencfg['samps_per_clk']
    fclk = soccfg['gens'][ch]['f_fabric']
    time_range = t_block[-1] - t_block[0]
    tot_cycles = int(np.round(time_range * 1000000 * fclk))
    block_cycles = samps_per_clk * tot_cycles
    block_us_arr = np.linspace(0, block_cycles / (1000000 * samps_per_clk * fclk), block_cycles)
    return amp_func(block_us_arr)


def plot_pulse_hist(time_hist, amp_hist):
    for i in range(len(time_hist)):
        t_block_mat = time_hist[i]
        amps = amp_hist[i]
        t_arr = np.linspace(t_block_mat[0][0], t_block_mat[0][-1], 501)
        # i_amps = amp_hist[i][0]
        # q_amps = amp_hist[i][1]

        drive_amp_funcs = [
            [rect_seg(amps[k][j], t_block_mat[k][j], t_block_mat[k][j + 1]) for j in range(len(t_block_mat[k]) - 1)] for
            k in range(len(t_block_mat))]

        comp_amp_funcs = [func_sum(np.array(drive_amp_funcs[i])) for i in range(len(t_block_mat))]

        for j in range(int(len(t_block_mat) / 2)):
            plt.plot(t_arr, comp_amp_funcs[j * 2](t_arr), label=f'Drive {j}i Iter: {i}', color=colors(i),
                     linestyle="solid")
            plt.plot(t_arr, comp_amp_funcs[(j * 2) + 1](t_arr), label=f'Drive {j}q Iter: {i}', color=colors(i),
                     linestyle="dotted")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.title("Drive pulses by iteration")
    plt.legend()
    plt.show()


def plot_cost_iters_R(time_hist, amp_hist, freq_hist, cost_func, *args):
    costs = []
    for i in range(len(time_hist)):
        args = (args[0], i + 1, time_hist[i], args[3], args[4], args[5], args[6], args[7])
        x0 = np.append(amp_hist[i].flatten(), freq_hist[i])
        costs.append(cost_func(x0, *args))
    plt.plot(costs, marker=".")
    plt.xlabel("Segments")
    plt.ylabel("Cost")
    plt.title("Optimization cost by segments")
    plt.show()


def plot_cost_iters(cost_hist):
    plt.plot(np.arange(1, len(cost_hist) + 1, 1), cost_hist, marker=".")
    plt.xlabel("Segments")
    plt.ylabel("Cost")
    plt.title("Optimization cost by segments")
    plt.show()


def post_opt_drives(opts, *args):
    num_drives = args[0]
    num_segs = args[1]
    t_block_mat = args[2]
    i_amps = np.round(opts[:num_segs])
    q_amps = np.round(opts[num_segs:2 * num_segs])
    freqs = opts[-num_drives:]
    print(f'Trial I amps: {i_amps}')
    print(f'Trial Q amps: {q_amps}')
    print(f'Trial freqs: {freqs}')
    t_arr = np.linspace(t_block_mat[0][0], t_block_mat[0][-1], 501)

    qubit_I_drive_amp_funcs = [rect_seg(i_amps[i], t_block_mat[0][i], t_block_mat[0][i + 1]) for i in
                               range(len(t_block_mat[0]) - 1)]
    qubit_Q_drive_amp_funcs = [rect_seg(q_amps[i], t_block_mat[1][i], t_block_mat[1][i + 1]) for i in
                               range(len(t_block_mat[1]) - 1)]

    qubit_I_drive_amps = func_sum(np.array(qubit_I_drive_amp_funcs))
    qubit_Q_drive_amps = func_sum(np.array(qubit_Q_drive_amp_funcs))

    plt.plot(t_arr, qubit_I_drive_amps(t_arr), label="Qubit I")
    plt.plot(t_arr, qubit_Q_drive_amps(t_arr), label="Qubit Q")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.title("Drive pulses")
    plt.legend()
    plt.show()

# expands the amplitude matrix, duplicating values for in between time steps allowing for time independent simulation
def ti_sim_seg_split(time_mat, amp_mat):
    # finds all unique times in the time matrix
    times = np.sort(np.unique(time_mat.flatten()))
    # create a new matrix to store new amplitudes in
    extend_amp_mat = np.empty((np.shape(amp_mat)[0], np.shape(times)[0] - 1))
    # Duplicate some amplitudes and add them to the new matrix
    for i, time in enumerate(times[:-1]):
        # print(f'time: {time}')
        # print(f'time_mat[{j}][:-1]: {time_mat[j][:-1]}')
        # print(f'amp_mat[{j}]: {amp_mat[j]}')
        extend_amp_mat[:, i] = [amp_by_time(time, time_mat[j][:-1], amp_mat[j]) for j in range(np.shape(time_mat)[0])]
    return times, extend_amp_mat

#
def iq_to_damps(i_amp, q_amp):
    return np.sqrt((i_amp**2) + (q_amp**2)) * np.arctan2(q_amp, i_amp)

#
def assemble_ti_hamil(h0, ops, amps):
    for i in range(len(ops)):
        h0 = h0 + (ops[i] * amps[i])
    return h0

def build_psi(state_sizes, init_state):
    basis_states = []
    for i in range(len(state_sizes)):
        basis_states.append(basis(state_sizes[i], init_state[i]))
    return tensor(basis_states)

def iqs_to_camps(damps):
    camp_damps = []
    for dr in range(int(np.shape(damps)[0] / 2)):
        camp_damps.append(iq_to_amp(damps[2 * dr], damps[(2 * dr) + 1]))
        camp_damps.append(iq_to_amp(damps[2 * dr], damps[(2 * dr) + 1], conj=True))
    camp_damps = np.array(camp_damps)
    return camp_damps


def piecewise_mesolve(H_r, rho0, time_splits, tstep, drive_ops, drive_camps, c_ops, expect_ops, options,
                      save_hist=False):
    sim_states = 0
    curr_state = 0
    if save_hist:
        save_times = np.array([0.0])
        sim_states = np.array([rho0.full()])
        save_states = []
        # Populating the first index of v
        for i in range(len(expect_ops)):
            save_states.append((rho0.dag() * expect_ops[i] * rho0).full())
        save_states = np.array([save_states]).T
    else:
        curr_state = rho0

    for i in range(len(time_splits) - 1):

        # Find the time range to evolve over
        ti = time_splits[i]
        tf = time_splits[i + 1]
        tarr = np.linspace(ti, tf, int((tf - ti) / tstep) + 1)

        # Add in drive terms
        H_s = assemble_ti_hamil(H_r, drive_ops, drive_camps[:, i])

        if save_hist:
            # Run simulation
            res = mesolve(H_s, sim_states[-1], tarr, c_ops, expect_ops, options=options)

            # Save results of evolution step
            save_times = np.append(save_times, tarr[1:])
            save_states = np.append(save_states, np.array(res.expect)[:, 1:], axis=1)
            sim_states = np.append(sim_states, res.states[-1])
        else:
            res = mesolve(H_s, curr_state, tarr, c_ops, expect_ops, options=options)
            curr_state = res.states[-1]

    if save_hist:
        return save_times, save_states
    else:
        return np.array(res.expect)[:, -1], curr_state


def ti_state_plot(opts, *args):
    num_drives = args[0]
    num_elems = args[1]
    drive_ops = args[2]
    t_block_mat = args[3]
    t_step = args[4]
    H_0 = args[5]
    init_state = args[6]
    c_ops = args[7]
    eval_ops = args[8]
    options = args[9]
    element_ops = args[10]
    element_freqs = args[11]
    output_cost_func = args[12]
    verbose = args[13]
    if num_elems == 0:
        amp_mat = np.reshape(opts, (num_drives * 2, int(len(opts) / (2 * num_drives))))
        drive_freqs = []
    else:
        amp_mat = np.reshape(opts[:-num_elems], (num_drives * 2, int((len(opts) - num_elems) / (2 * num_drives))))
        drive_freqs = opts[-num_elems:]
    if verbose:
        for i in range(num_drives):
            print(f'Trial I amp {i}: {amp_mat[2 * i]}')
            print(f'Trial Q amp {i}: {amp_mat[(2 * i) + 1]}')
        for i in range(num_elems):
            print(f'Trial freq {i}: {drive_freqs[i]}')

    # Convert to numpy arrays:
    # drive_ops = np.array(drive_ops)
    drive_freqs = np.array(drive_freqs)
    # element_ops = np.array(element_ops)
    element_freqs = np.array(element_freqs)

    split_times, split_amps = ti_sim_seg_split(t_block_mat, amp_mat)

    drive_op_amps = iqs_to_camps(split_amps) * 2 * np.pi

    # Setting up
    H_r = assemble_ti_hamil(H_0, element_ops, (element_freqs - drive_freqs))

    options.store_states = True

    # Simulate
    save_times, save_states = piecewise_mesolve(H_r, init_state, split_times, t_step, drive_ops, drive_op_amps, c_ops,
                                                eval_ops, options, True)

    for i in range(len(save_states)):
        plt.plot(save_times, np.abs(save_states[i]), label=f'Eval state: {i + 1}')
    plt.xlabel("Times (s)")
    plt.ylabel("Occupation")
    plt.title("Evolution of tracked states")
    plt.show()

def sim_interp_cost_eval(opts, *args):
    if len(args) == 1:
        args = args[0]
    num_drives = args[0]
    drive_ops = args[1]
    freqs = args[2]
    H_0 = args[3]
    init_state = args[4]
    eval_ops = args[5]
    options = args[6]
    output_cost_func = args[7]
    time_start = args[8]
    time_stop = args[9]
    drive_freqs = args[10]
    drive_elem_nums = args[11]
    verbose = args[12]
    plot_trial_pulses = args[13]

    t_arr = np.linspace(time_start, time_stop, 1001)

    amps = opts[:int(len(opts) / 2)]
    times = opts[int(len(opts) / 2):]

    for i in range(len(times)):
        if (times[i] <= time_start) or (times[i] >= time_stop):
            print(f'Times out of bounds: {times[i]}')
            return 0

    amps = np.round(np.reshape(amps, (num_drives * 2, int(len(amps) / (num_drives * 2)))))
    t_block_mat = np.reshape(times, (num_drives * 2, int(len(times) / (num_drives * 2))))

    reorder = t_block_mat.argsort()

    for i in range(np.shape(amps)[0]):
        amps[i] = amps[i][reorder[i]]
        t_block_mat[i] = t_block_mat[i][reorder[i]]

    start_buffer_arr = time_start * np.ones((2 * num_drives, 1), dtype=np.float32)
    stop_buffer_arr = time_stop * np.ones((2 * num_drives, 1), dtype=np.float32)

    t_block_mat = np.append(np.append(start_buffer_arr, t_block_mat, axis=1), stop_buffer_arr, axis=1)

    amps = interp_amp_wrapper(amps)

    if verbose:
        for i in range(num_drives):
            print(f'Trial I time {i}: {t_block_mat[2 * i]}')
            print(f'Trial I amp {i}: {amps[2 * i]}')
            print(f'Trial Q time {i}: {t_block_mat[(2 * i) + 1]}')
            print(f'Trial Q amp {i}: {amps[(2 * i) + 1]}')

    # Returns a cost of 0 if any times overlap
    for i in range(np.shape(amps)[0]):
        for j in range(np.shape(amps)[1] - 1):
            if t_block_mat[i, j] == t_block_mat[i, j + 1]:
                print(f'Times overlap: {t_block_mat[i, j]} and {t_block_mat[i, j + 1]}')
                return 0

    drive_amp_funcs = [
        [interp_seg(amps[j][i:i + 2], t_block_mat[j][i:i + 2]) for i in range(len(t_block_mat[j]) - 1)] for j
        in range(num_drives * 2)]

    comp_amp_funcs = [func_sum(np.array(drive_amp_funcs[i])) for i in range(num_drives * 2)]

    # Plot out the amplitudes of the drives
    if plot_trial_pulses:
        t_arr = np.linspace(0, t_block_mat[0][-1], 501)
        for i in range(num_drives):
            plt.plot(t_arr, comp_amp_funcs[(2 * i)](t_arr), label=f'Drive Re({i})', color=colors(i), linestyle="solid")
            plt.plot(t_arr, comp_amp_funcs[(2 * i) + 1](t_arr), label=f'Drive Im({i})', color=colors(i),
                     linestyle="dotted")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.title("Drive pulses")
        plt.legend()
        save_dir = r'C:\_Data\images\20241208'
        img_name = time.strftime('%Y%m%d-%H%M%S.png')
        plt.savefig(os.path.join(save_dir, img_name))
        print(f'Saved pulse plot to {os.path.join(save_dir, img_name)}')
        plt.close()

    drive_camps = []
    for i in range(0, 2 * num_drives, 2):
        drive_camps.append(iq_to_amp(comp_amp_funcs[i], comp_amp_funcs[i + 1]))
        drive_camps.append(iq_to_amp(comp_amp_funcs[i], comp_amp_funcs[i + 1], conj=True))

    drive_osc_camps = []

    for i in range(num_drives * 2):
        drive_osc_camps = [drive_osc_camp(drive_camps[i], zero, freqs[drive_elem_nums[floor(i / 2)]], drive_freqs[floor(i / 2)]) for i in
                       range(num_drives * 2)]

    if plot_trial_pulses:
        t_arr = np.linspace(0, t_block_mat[0][-1], 501)
        for i in range(num_drives):
            plt.plot(t_arr, np.abs(drive_osc_camps[(2 * i)](t_arr)), label=f'Drive Re({i})', color=colors(i), linestyle="solid")
            plt.plot(t_arr, np.abs(drive_osc_camps[(2 * i) + 1](t_arr)), label=f'Drive Im({i})', color=colors(i),
                     linestyle="dotted")
            print(comp_amp_funcs[(2 * i)](t_arr)[0])
            print(comp_amp_funcs[(2 * i) + 1](t_arr)[0])
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.title("Drive pulses")
        plt.legend()
        save_dir = r'C:\_Data\images\20241205'
        img_name = time.strftime('%Y%m%d-%H%M%S.png')
        plt.savefig(os.path.join(save_dir, img_name))
        plt.close()

    H_t = [[drive_ops[i], drive_osc_camps[i]] for i in range(len(drive_osc_camps))]

    H = [H_0] + H_t
    H = QobjEvo(H, tlist=t_arr)

    res = mesolve(H, init_state, t_arr, c_ops=[], e_ops=eval_ops, options=options)

    final_expect = res.expect

    final_dm = res.final_state

    if verbose:
        print(f'final_dm: {final_dm}')

    cost = np.abs(output_cost_func(final_expect, final_dm))

    if verbose:
        print(f'Cost: {cost}')

    return cost

def sim_interp_cost_eval_no_time(opts, *args):
    if len(args) == 1:
        args = args[0]
    num_drives = args[0]
    drive_ops = args[1]
    freqs = args[2]
    H_0 = args[3]
    init_state = args[4]
    eval_ops = args[5]
    options = args[6]
    output_cost_func = args[7]
    time_start = args[8]
    time_stop = args[9]
    drive_freqs = args[10]
    drive_elem_nums = args[11]
    verbose = args[12]
    plot_trial_pulses = args[13]
    times = args[14]

    options.nsteps = 10000

    t_arr = np.linspace(time_start, time_stop, 1001)

    amps = opts

    amps = np.round(np.reshape(amps, (num_drives * 2, int(len(amps) / (num_drives * 2)))))
    t_block_mat = np.reshape(times, (num_drives * 2, int(len(times) / (num_drives * 2))))

    start_buffer_arr = time_start * np.ones((2 * num_drives, 1), dtype=np.float32)
    stop_buffer_arr = time_stop * np.ones((2 * num_drives, 1), dtype=np.float32)

    t_block_mat = np.append(np.append(start_buffer_arr, t_block_mat, axis=1), stop_buffer_arr, axis=1)

    amps = interp_amp_wrapper(amps)

    if verbose:
        for i in range(num_drives):
            print(f'Trial I time {i}: {t_block_mat[2 * i]}')
            print(f'Trial I amp {i}: {amps[2 * i]}')
            print(f'Trial Q time {i}: {t_block_mat[(2 * i) + 1]}')
            print(f'Trial Q amp {i}: {amps[(2 * i) + 1]}')

    drive_amp_funcs = [
        [interp_seg(amps[j][i:i + 2], t_block_mat[j][i:i + 2]) for i in range(len(t_block_mat[j]) - 1)] for j
        in range(num_drives * 2)]

    comp_amp_funcs = [func_sum(np.array(drive_amp_funcs[i])) for i in range(num_drives * 2)]

    # Plot out the amplitudes of the drives
    if plot_trial_pulses:
        t_arr = np.linspace(0, t_block_mat[0][-1], 501)
        for i in range(num_drives):
            plt.plot(t_arr, comp_amp_funcs[(2 * i)](t_arr), label=f'Drive Re({i})', color=colors(i), linestyle="solid")
            plt.plot(t_arr, comp_amp_funcs[(2 * i) + 1](t_arr), label=f'Drive Im({i})', color=colors(i),
                     linestyle="dotted")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.title("Drive pulses")
        plt.legend()
        save_dir = r'C:\_Data\images\20241205'
        img_name = time.strftime('%Y%m%d-%H%M%S.png')
        plt.savefig(os.path.join(save_dir, img_name))
        plt.close()

    drive_camps = []
    for i in range(0, 2 * num_drives, 2):
        drive_camps.append(iq_to_amp(comp_amp_funcs[i], comp_amp_funcs[i + 1]))
        drive_camps.append(iq_to_amp(comp_amp_funcs[i], comp_amp_funcs[i + 1], conj=True))

    drive_osc_camps = []

    for i in range(num_drives * 2):
        drive_osc_camps = [drive_osc_camp(drive_camps[i], zero, freqs[drive_elem_nums[floor(i / 2)]], drive_freqs[floor(i / 2)]) for i in
                       range(num_drives * 2)]

    H_t = [[drive_ops[i], drive_osc_camps[i]] for i in range(len(drive_osc_camps))]

    H = [H_0] + H_t
    H = QobjEvo(H, tlist=t_arr)

    res = mesolve(H, init_state, t_arr, c_ops=[], e_ops=eval_ops, options=options)

    final_expect = res.expect

    final_dm = res.final_state

    if verbose:
        print(f'final_dm: {final_dm}')

    cost = np.abs(output_cost_func(final_expect, final_dm))

    if verbose:
        print(f'Cost: {cost}')

    return cost

def ti_cost_eval(opts, *args):
    num_drives = args[0]
    num_elems = args[1]
    drive_ops = args[2]
    t_block_mat = args[3]
    t_step = args[4]
    H_0 = args[5]
    init_state = args[6]
    c_ops = args[7]
    eval_ops = args[8]
    options = args[9]
    element_ops = args[10]
    element_freqs = args[11]
    output_cost_func = args[12]
    verbose = args[13]
    if num_elems == 0:
        amp_mat = np.reshape(opts, (num_drives * 2, int(len(opts) / (2 * num_drives))))
        drive_freqs = []
    else:
        amp_mat = np.reshape(opts[:-num_elems], (num_drives * 2, int((len(opts) - num_elems) / (2 * num_drives))))
        drive_freqs = opts[-num_elems:]
    if verbose:
        for i in range(num_drives):
            print(f'Trial I amp {i}: {amp_mat[2 * i]}')
            print(f'Trial Q amp {i}: {amp_mat[(2 * i) + 1]}')
        for i in range(num_elems):
            print(f'Trial freq {i}: {drive_freqs[i]}')

    # Convert to numpy arrays:
    # drive_ops = np.array(drive_ops)
    drive_freqs = np.array(drive_freqs)
    # element_ops = np.array(element_ops)
    element_freqs = np.array(element_freqs)

    split_times, split_amps = ti_sim_seg_split(t_block_mat, amp_mat)

    drive_op_amps = iqs_to_camps(split_amps) * 2 * np.pi

    # Setting up Hamiltonain
    H_r = assemble_ti_hamil(H_0, element_ops, (element_freqs - drive_freqs))

    options.store_states = True
    # options["rhs_reuse"] = False

    # Simulate
    final_expect, final_dm = piecewise_mesolve(H_r, init_state, split_times, t_step, drive_ops, drive_op_amps, c_ops, eval_ops,
                                   options, False)

    cost = np.abs(output_cost_func(final_expect, final_dm))

    # clear_output(wait=False)

    # cost = round(cost, 6)  # + np.abs(np.sum(np.sqrt(np.power(i_amps, 2) + np.power(q_amps, 2))) / 100000000)
    if verbose:
        print(f'Cost: {cost}')

    return cost


# def sim_interp_cost_eval(opts, *args):
#     num_drives = args[0]
#     drive_ops = args[1]
#     w0_arr = args[2]
#     H_0 = args[3]
#     init_state = args[4]
#     t_arr = args[5]
#     eval_ops = args[6]
#     options = args[7]
#     output_cost_func = args[8]
#     verbose = args[9]
#     drive_freqs = args[10]
#     drive_elem_nums = args[11]
#     # print(f'drive_freqs: {drive_freqs}')
#
#     # amps = np.round(np.reshape(amps, (num_drives * 2, int(len(amps) / (num_drives * 2)))))
#     # freqs = opts[-num_drives:]
#
#     time_start = t_arr[0]
#     time_stop = t_arr[-1]
#     amps_times = opts[:-num_drives]
#     amps = amps_times[:int(len(amps_times) / 2)]
#     times = amps_times[int(len(amps_times) / 2):]
#     freqs = w0_arr
#
#     for i in range(len(times)):
#         if (times[i] <= time_start) or (times[i] >= time_stop):
#             return 0
#
#     amps = np.round(np.reshape(amps, (num_drives * 2, int(len(amps) / (num_drives * 2)))))
#     t_block_mat = np.reshape(times, (num_drives * 2, int(len(times) / (num_drives * 2))))
#
#     reorder = t_block_mat.argsort()
#
#     for i in range(np.shape(amps)[0]):
#         amps[i] = amps[i][reorder[i]]
#         t_block_mat[i] = t_block_mat[i][reorder[i]]
#
#     start_buffer_arr = time_start * np.ones((2 * num_drives, 1), dtype=np.float32)
#     stop_buffer_arr = time_stop * np.ones((2 * num_drives, 1), dtype=np.float32)
#
#     t_block_mat = np.append(np.append(start_buffer_arr, t_block_mat, axis=1), stop_buffer_arr, axis=1)
#
#     amps = interp_amp_wrapper(amps)
#
#     if verbose:
#         for i in range(num_drives):
#             print(f'Trial I time {i}: {t_block_mat[2 * i]}')
#             print(f'Trial I amp {i}: {amps[2 * i]}')
#             print(f'Trial Q time {i}: {t_block_mat[(2 * i) + 1]}')
#             print(f'Trial Q amp {i}: {amps[(2 * i) + 1]}')
#             print(f'Trial freq {i}: {drive_freqs[i]}')
#
#     # Returns a cost of 0 if any times overlap
#     for i in range(np.shape(amps)[0]):
#         for j in range(np.shape(amps)[1] - 1):
#             if t_block_mat[i, j] == t_block_mat[i, j + 1]:
#                 print(f'Times overlap: {t_block_mat[i, j]} and {t_block_mat[i, j + 1]}')
#                 return 0
#
#     # drive_amp_funcs = []
#
#     # for i in range(num_drives * 2):
#     #     t_arr = []
#     #     for j in range(len(t_block_mat[j]) - 1):
#     #         t_arr.append(interp_seg(amps[j][i:i + 2], t_block_mat[j][i:i + 2]))
#
#
#     drive_amp_funcs = [
#         [interp_seg(amps[j][i:i + 2], t_block_mat[j][i:i + 2]) for i in range(len(t_block_mat[j]) - 1)] for j
#         in range(num_drives * 2)]
#
#     comp_amp_funcs = [func_sum(np.array(drive_amp_funcs[i])) for i in range(num_drives * 2)]
#
#     drive_camps = []
#     for i in range(0, 2 * num_drives, 2):
#         drive_camps.append(iq_to_amp(comp_amp_funcs[i], comp_amp_funcs[i + 1]))
#         drive_camps.append(iq_to_amp(comp_amp_funcs[i], comp_amp_funcs[i + 1], conj=True))
#
#     drive_osc_camps = []
#
#     for i in range(num_drives * 2):
#         drive_osc_camps = [drive_osc_camp(drive_camps[i], zero, w0_arr[drive_elem_nums[floor(i / 2)]], drive_freqs[floor(i / 2)]) for i in
#                        range(num_drives * 2)]
#
#     H_t = [[drive_ops[i], drive_osc_camps[i]] for i in range(len(drive_osc_camps))]
#
#     H = [H_0] + H_t
#     H = QobjEvo(H, tlist=t_arr)
#
#     res = mesolve(H, init_state, t_arr, c_ops=[], e_ops=eval_ops, options=options)
#
#     final_expect = res.expect
#
#     final_dm = res.final_state
#
#     if verbose:
#         print(f'final_dm: {final_dm}')
#
#     cost = np.abs(output_cost_func(final_expect, final_dm))
#
#     if verbose:
#         print(f'Cost: {cost}')
#
#     return cost


def sim_interp_ti_cost_eval(opts, *args):
    num_drives = args[0]
    drive_ops = args[1]
    w0_arr = args[2]
    H_0 = args[3]
    init_state = args[4]
    t_arr = args[5]
    eval_ops = args[6]
    options = args[7]
    output_cost_func = args[8]
    verbose = args[9]
    element_ops = args[10]
    t_step = args[11]

    time_start = t_arr[0]
    time_stop = t_arr[-1]
    amps_times = opts[:-num_drives]
    amps = amps_times[:int(len(amps_times) / 2)]
    times = amps_times[int(len(amps_times) / 2):]
    freqs = w0_arr

    amps = np.round(np.reshape(amps, (num_drives * 2, int(len(amps) / (num_drives * 2)))))
    t_block_mat = np.reshape(times, (num_drives * 2, int(len(times) / (num_drives * 2))))

    reorder = t_block_mat.argsort()

    for i in range(np.shape(amps)[0]):
        amps[i] = amps[i][reorder[i]]
        t_block_mat[i] = t_block_mat[i][reorder[i]]

    start_buffer_arr = time_start * np.ones((2 * num_drives, 1), dtype=np.float32)
    stop_buffer_arr = time_stop * np.ones((2 * num_drives, 1), dtype=np.float32)

    t_block_mat = np.append(np.append(start_buffer_arr, t_block_mat, axis=1), stop_buffer_arr, axis=1)

    amps = interp_amp_wrapper(amps)

    # if num_drives == 0:
    #     amp_mat = np.reshape(opts, (num_drives * 2, int(len(opts) / (2 * num_drives))))
    #     drive_freqs = []
    # else:
    #     amp_mat = np.reshape(opts[:-num_elems], (num_drives * 2, int((len(opts) - num_elems) / (2 * num_drives))))
    #     drive_freqs = opts[-num_elems:]

    if verbose:
        for i in range(num_drives):
            print(f'Trial I time {i}: {t_block_mat[2 * i]}')
            print(f'Trial I amp {i}: {amps[2 * i]}')
            print(f'Trial Q time {i}: {t_block_mat[(2 * i) + 1]}')
            print(f'Trial Q amp {i}: {amps[(2 * i) + 1]}')
            print(f'Trial freq {i}: {freqs[i]}')

    # Convert to numpy arrays:
    # drive_ops = np.array(drive_ops)
    freqs = np.array(freqs)
    # element_ops = np.array(element_ops)
    freqs = np.array(freqs)

    split_times, split_amps = ti_sim_seg_split(t_block_mat, amps)

    drive_op_amps = iqs_to_camps(split_amps) * 2 * np.pi

    # Setting up Hamiltonian
    H_r = assemble_ti_hamil(H_0, element_ops, (freqs - freqs))

    options.store_states = True
    # options["rhs_reuse"] = False

    # Simulate
    final_expect, final_dm = piecewise_mesolve(H_r, init_state, split_times, t_step, drive_ops, drive_op_amps, c_ops, eval_ops,
                                   options, False)

    cost = np.abs(output_cost_func(final_expect, final_dm))

    # clear_output(wait=False)

    # cost = round(cost, 6)  # + np.abs(np.sum(np.sqrt(np.power(i_amps, 2) + np.power(q_amps, 2))) / 100000000)
    if verbose:
        print(f'Cost: {cost}')

    return cost


def cost_qubit_x(final_state):
    return 1 - final_state[0]

def drive_pulse_plot(times, amps, legend=True):
    split_times, split_amps = ti_sim_seg_split(times, amps)
    t_arr = np.linspace(split_times[0], split_times[-1], 501)
    for i in range(int(len(split_amps) / 2)):
        plt.plot(t_arr, np.vectorize(amp_by_time, excluded=(1, 2))(t_arr, split_times, split_amps[2 * i]) * 2 * np.pi, label=f'Drive {i} I', color=colors(i))
        plt.plot(t_arr, np.vectorize(amp_by_time, excluded=(1, 2))(t_arr, split_times, split_amps[(2 * i) + 1]) * 2 * np.pi, label=f'Drive {i} Q', color=colors(i),linestyle="dotted")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (Hz * 2pi)")
    plt.title("Optimal pulse amplitudes")
    if legend:
        plt.legend()
    plt.show()

def drive_pulse_hist_plot(time_hist, amp_hist, legend=True):
    for i, amps in enumerate(amp_hist):
        split_times, split_amps = ti_sim_seg_split(time_hist[i], amps)
        t_arr = np.linspace(split_times[0], split_times[-1], 501)
        for j in range(int(len(split_amps) / 2)):
            plt.plot(t_arr, np.vectorize(amp_by_time, excluded=(1, 2))(t_arr, split_times, split_amps[2 * j]) * 2 * np.pi, label=f'Split {i}, Drive {j} I', color=colors(i), linestyle=linestyles(j))
            plt.plot(t_arr, np.vectorize(amp_by_time, excluded=(1, 2))(t_arr, split_times, split_amps[(2 * j) + 1]) * 2 * np.pi, label=f'Split {i}, Drive {j} Q', color=colors(i),linestyle=linestyles(j + 5))
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (Hz * 2pi)")
    plt.title("Optimal pulse amplitudes for # of splits")
    if legend:
        plt.legend()
    plt.show()

def exp_interp_shape_plot(opts, *args):
    drive_chs = args[0]
    plot_opt_pulses = args[2]
    time_start = args[4]
    time_stop = args[5]

    num_drives = len(drive_chs)
    amps_times = opts[:-num_drives]
    amps = amps_times[:int(len(amps_times) / 2)]
    times = amps_times[int(len(amps_times) / 2):]
    freqs = opts[-num_drives:]

    amps = np.round(np.reshape(amps, (num_drives * 2, int(len(amps) / (num_drives * 2)))))
    t_block_mat = np.reshape(times, (num_drives * 2, int(len(times) / (num_drives * 2))))

    reorder = t_block_mat.argsort()

    for i in range(np.shape(amps)[1]):
        amps[i] = amps[i][reorder[i]]
        t_block_mat[i] = t_block_mat[i][reorder[i]]

    start_buffer_arr = time_start * np.ones((2 * num_drives, 1), dtype=np.float32)
    stop_buffer_arr = time_stop * np.ones((2 * num_drives, 1), dtype=np.float32)

    t_block_mat = np.append(np.append(start_buffer_arr, t_block_mat, axis=1), stop_buffer_arr, axis=1)

    amps = interp_amp_wrapper(amps)

    drive_amp_funcs = [
        [interp_seg(amps[j][i:i+2], t_block_mat[j][i:i+2]) for i in range(len(t_block_mat[j]) - 1)] for j
        in range(num_drives * 2)]

    comp_amp_funcs = [func_sum(np.array(drive_amp_funcs[i])) for i in range(num_drives * 2)]

    # Plot out the amplitudes of the drives
    t_arr = np.linspace(0, t_block_mat[0][-1], 501)
    for i in range(num_drives):
        plt.plot(t_arr, comp_amp_funcs[(2 * i)](t_arr), label=f'Drive Re({i})', color=colors(i), linestyle="solid")
        plt.plot(t_arr, comp_amp_funcs[(2 * i) + 1](t_arr), label=f'Drive Im({i})', color=colors(i),
                 linestyle="dotted")
        print(comp_amp_funcs[(2 * i)](t_arr)[0])
        print(comp_amp_funcs[(2 * i) + 1](t_arr)[0])
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.title("Drive pulses")
    plt.legend()
    plt.show()

def exp_interp_pulse_plot(opts, *args):
    # setup
    config_dict = pc.config_dict

    # Build local connection
    Pyro4.config.SERIALIZER = "pickle"
    Pyro4.config.PICKLE_PROTOCOL_VERSION = 4

    # IP address of the QICK board
    ns_host = "192.168.20.101"
    ns_port = 8888
    proxy_name = "myqick"

    ns = Pyro4.locateNS(host=ns_host, port=ns_port)

    soc = Pyro4.Proxy(ns.lookup(proxy_name))
    soccfg = QickConfig(soc.get_cfg())

    # actual code
    drive_chs = args[0]
    time_start = args[4]
    time_stop = args[5]

    num_drives = len(drive_chs)
    amps_times = opts[:-num_drives]
    amps = amps_times[:int(len(amps_times) / 2)]
    times = amps_times[int(len(amps_times) / 2):]
    freqs = opts[-num_drives:]

    amps = np.round(np.reshape(amps, (num_drives * 2, int(len(amps) / (num_drives * 2)))))
    t_block_mat = np.reshape(times, (num_drives * 2, int(len(times) / (num_drives * 2))))

    reorder = t_block_mat.argsort()

    for i in range(np.shape(amps)[1]):
        amps[i] = amps[i][reorder[i]]
        t_block_mat[i] = t_block_mat[i][reorder[i]]

    start_buffer_arr = time_start * np.ones((2 * num_drives, 1), dtype=np.float32)
    stop_buffer_arr = time_stop * np.ones((2 * num_drives, 1), dtype=np.float32)

    t_block_mat = np.append(np.append(start_buffer_arr, t_block_mat, axis=1), stop_buffer_arr, axis=1)

    amps = interp_amp_wrapper(amps)

    drive_amp_funcs = [
        [interp_seg(amps[j][i:i+2], t_block_mat[j][i:i+2]) for i in range(len(t_block_mat[j]) - 1)] for j
        in range(num_drives * 2)]

    comp_amp_funcs = [func_sum(np.array(drive_amp_funcs[i])) for i in range(num_drives * 2)]

    # qick program stuff
    config = copy.deepcopy(pc.config_dict)

    expt_dict = {
        "reps": 1000,  # inner repeat times
        "rounds": 1,  # outer loop times
        "soft_avg": 1,  # soft average (only for decimated readout)
        "expts": 1,  # parameters sweep
        "hdf5": "pi-opt",
    }
    config["expt"] = expt_dict

    measure_state = "g"
    storage = "B"

    # Modify eval_dict to work for this experiment
    config["eval"]["coeffs"] = [1, -1]
    config["eval"]["norm_amp"] = 1
    config["eval"]["background_amp"] = 0
    config["eval"]["measure_state"] = measure_state

    config["readout"]["relax_delay"] = 1000

    config["res"]["freq"] = config["res"]["freqs"][measure_state]

    observable_dicts = [
        [
            pc.pi_pulse(rabi_states="ef", meas_state=measure_state),
        ],
        [
            pc.pi_pulse(rabi_states="ef", meas_state=measure_state),
            pc.pi_pulse(rabi_states="ge", meas_state=measure_state, selective=True,
                        freq=config_dict["qubit"]["freqs"]["g"]["ge"] + config_dict["storage_A"][
                            "chi"] + config_dict["storage_B"]["chi"]),
        ],
    ]

    normVolt = ArbHamiltonianEval(soc=soc, soccfg=soccfg, config=config, observable_dicts=observable_dicts)

    # calculate drive amps for all times that the tprocessor will run for
    spread_drive_amps = []
    for i in range(num_drives):
        spread_drive_amps.append(
            qick_spread_pulse_amps(soccfg, t_block_mat[2 * i], comp_amp_funcs[2 * i], drive_chs[i]))
        spread_drive_amps.append(
            qick_spread_pulse_amps(soccfg, t_block_mat[(2 * i) + 1], comp_amp_funcs[(2 * i) + 1], drive_chs[i]))

    spread_drive_amps = np.round(spread_drive_amps).astype(int)

    for i, ch in enumerate(drive_chs):
        gencfg = soccfg['gens'][ch]

        normVolt.add_test_waveform(ch, {
            "name": f'test_wf_{i}',
            "I": spread_drive_amps[2 * i],
            "Q": spread_drive_amps[(2 * i) + 1],
            "gain": int(gencfg['maxv'] * gencfg['maxv_scale']),
            "freq": freqs[i],
            "phase": 0,
            "t": 'auto'  # soccfg.us2cycles(gen_ch=ch, us=0.05) + 16,
        })

    for i in range(len(observable_dicts)):
        normVolt.pulse_view(title="e EF contrast", n=i)

# Fits data to a sin curve TODO: have func return figure instead of instantly plotting
def sin_fit(x, y):
    def sin_helper(t, A, w, d, C):
        return (A * np.sin((t * w) - d)) + C

    min_amp = min(y)
    max_amp = max(y)
    mid_amp = (min_amp + max_amp) / 2
    disp_amp = max_amp - mid_amp

    p0 = [disp_amp, 2 * np.pi / (x[-1] - x[0]), np.arctan2(y[0], disp_amp), mid_amp]

    popt, pcorr = curve_fit(sin_helper, x, y, p0=p0)

    print(f'popt: {popt}')

    print(f'pi_amp: {round(2 * np.pi / popt[1])}')

    plt.plot(x, y, label="Data")
    plt.plot(x, sin_helper(x, *popt), label="Fit")
    plt.legend()
    plt.show()

def SSR_map(i_vals, q_vals, g_blob, e_blob):
    i_vals = np.array(i_vals)
    q_vals = np.array(q_vals)

    gd_i = i_vals - g_blob[0]
    gd_q = q_vals - g_blob[1]

    ed_i = i_vals - e_blob[0]
    ed_q = q_vals - e_blob[1]

    g_dist = np.sqrt(np.power(gd_i, 2) + np.power(gd_q, 2))
    e_dist = np.sqrt(np.power(ed_i, 2) + np.power(ed_q, 2))

    ret_arr = np.zeros(len(i_vals))

    ret_arr[e_dist < g_dist] = 1

    return ret_arr

def rabi_fit(params, x, data):
    '''

    parameters:
       ofs
       amp
       period
    '''

    cosine = np.cos(2 * np.pi * x / params['period'].value)

    est = params['ofs'].value + params['amp'].value * cosine
    return data - est