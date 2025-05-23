# Author: Ben Brock 
# Created on May 03, 2023 

import utils.pulse_configs as pc
from utils.opt_utils import *
from utils.hamiltonians import ArbHamiltonianEval
from utils.operations import *
from qick import QickConfig
import Pyro4
import copy
import csv

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

def time_eval(opts, *args):
    args = args[0]
    drive_chs = args[0]
    verbose = args[1]
    plot_opt_pulses = args[2]
    plot_pulse_viewer = args[3]
    time_start = args[4]
    time_stop = args[5]
    freqs = args[6]
    hdf5_file_path = args[7]

    num_drives = len(drive_chs)
    amps = opts[:int(len(opts) / 2)]
    times = opts[int(len(opts) / 2):]

    # Returns 0 if any times are outside the set bounds
    for i in range(len(times)):
        if (times[i] <= time_start) or (times[i] >= time_stop):
            print("Time out of bounds")
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

    # if verbose:
    #     for i in range(num_drives):
            # print(f'Trial I time {i}: {t_block_mat[2 * i]}')
            # print(f'Trial I amp {i}: {amps[2 * i]}')
            # print(f'Trial Q time {i}: {t_block_mat[(2 * i) + 1]}')
            # print(f'Trial Q amp {i}: {amps[(2 * i) + 1]}')

    # Returns a cost of 0 if any times overlap
    for i in range(np.shape(amps)[0]):
        for j in range(np.shape(amps)[1] - 1):
            if t_block_mat[i, j] == t_block_mat[i, j + 1]:
                print(f'Times overlap: {t_block_mat[i, j]} and {t_block_mat[i, j + 1]}')
                return 0

    drive_amp_funcs = [
        [interp_seg(amps[j][i:i+2], t_block_mat[j][i:i+2]) for i in range(len(t_block_mat[j]) - 1)] for j
        in range(num_drives * 2)]

    comp_amp_funcs = [func_sum(np.array(drive_amp_funcs[i])) for i in range(num_drives * 2)]

    # Plot out the amplitudes of the drives
    if plot_opt_pulses:
        t_arr = np.linspace(0, t_block_mat[0][-1], 501)
        for i in range(num_drives):
            plt.plot(t_arr, comp_amp_funcs[(2 * i)](t_arr), label=f'Drive Re({i})', color=colors(i), linestyle="solid")
            plt.plot(t_arr, comp_amp_funcs[(2 * i) + 1](t_arr), label=f'Drive Im({i})', color=colors(i),
                     linestyle="dotted")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.title("Drive pulses")
        plt.legend()
        Operations.save_figure(name="pre-comp-pulses")
        # plt.show()

    # qick program stuff
    config = copy.deepcopy(pc.config_dict)

    expt_dict = {
        "reps": 7500,  # inner repeat times
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
    # config["eval"]["norm_amp"] = 1
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
                        freq=config_dict["qubit"]["selective_freqs"]["g"]["ge"] + config_dict["storage_A"][
                            "chi"]),
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

    if plot_pulse_viewer:
        for i in range(len(observable_dicts)):
            normVolt.pulse_view(title="e EF contrast", n=i)

    q_amp = normVolt.expect(progress=False, self_project=True, save=False)

    cost = round(q_amp, 10)  # + np.abs(np.sum(np.sqrt(np.power(i_amps, 2) + np.power(q_amps, 2))) / 100000000)
    if verbose:
        print(f'Previous cost: {np.abs(cost)}')

    with open(hdf5_file_path, "a") as f:
        entrywriter = csv.writer(f, delimiter=",")
        entrywriter.writerow(np.append(np.append(t_block_mat, amps), np.abs(cost)).tolist())
        # f.write([t_block_mat, amps, np.abs(cost)])

    return np.abs(cost)