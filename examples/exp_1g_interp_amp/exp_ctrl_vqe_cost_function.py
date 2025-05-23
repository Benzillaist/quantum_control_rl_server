# Author: Ben Brock 
# Created on May 03, 2023 

import utils.pulse_configs as pc
from utils.opt_utils import *
from utils.hamiltonians import ArbHamiltonianEval
from utils.operations import *
from qick import QickConfig
import Pyro4
import copy
import json
import utils.pulse_elements as pe

# config_dict = pc.config_dict

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



def exp_g1_cost_func_x(opts, *args):
    drive_chs = args[0]
    verbose = args[1]
    plot_opt_pulses = args[2]
    plot_pulse_viewer = args[3]
    time_start = args[4]
    time_stop = args[5]
    freqs = args[6]
    times = args[7]

    # load config dict:
    with open("../../utils/pulse_configs.py") as f:
        data = f.read()

    config = json.loads(data)


    num_drives = len(drive_chs)
    amps = opts

    amps = np.round(np.reshape(amps, (num_drives * 2, int(len(amps) / (num_drives * 2)))))
    times = np.reshape(times, (num_drives * 2, int(len(times) / (num_drives * 2))))

    start_buffer_arr = time_start * np.ones((2 * num_drives, 1), dtype=np.float32)
    stop_buffer_arr = time_stop * np.ones((2 * num_drives, 1), dtype=np.float32)

    times = np.append(np.append(start_buffer_arr, times, axis=1), stop_buffer_arr, axis=1)

    amps = interp_amp_wrapper(amps)

    # if verbose:
    #     for i in range(num_drives):
    #         print(f'Trial I time {i}: {times[2 * i]}')
    #         print(f'Trial I amp {i}: {amps[2 * i]}')
    #         print(f'Trial Q time {i}: {times[(2 * i) + 1]}')
    #         print(f'Trial Q amp {i}: {amps[(2 * i) + 1]}')

    drive_amp_funcs = [
        [interp_seg(amps[j][i:i+2], times[j][i:i+2]) for i in range(len(times[j]) - 1)] for j
        in range(num_drives * 2)]

    comp_amp_funcs = [func_sum(np.array(drive_amp_funcs[i])) for i in range(num_drives * 2)]

    # Plot out the amplitudes of the drives
    if plot_opt_pulses:
        t_arr = np.linspace(0, times[0][-1], 501)
        for i in range(num_drives):
            plt.plot(t_arr, comp_amp_funcs[(2 * i)](t_arr), label=f'Drive Re({i})', color=colors(i), linestyle="solid")
            plt.plot(t_arr, comp_amp_funcs[(2 * i) + 1](t_arr), label=f'Drive Im({i})', color=colors(i),
                     linestyle="dotted")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.title("Drive pulses")
        plt.legend()
        save_dir = r'C:\_Data\images\20241208'
        img_name = time.strftime('trial-pulse-%Y%m%d-%H%M%S.png')
        plt.savefig(os.path.join(save_dir, img_name))
        plt.close()

    expt_dict = {
        "reps": 2000,  # inner repeat times
        "rounds": 1,  # outer loop times
        "soft_avg": 1,  # soft average (only for decimated readout)
        "expts": 1,  # parameters sweep
        "hdf5": "pi-opt",
    }
    config["expt"] = expt_dict

    measure_state = "g"
    storage = "A"

    # Modify eval_dict to work for this experiment
    config["eval"]["coeffs"] = [1, -1]
    # config["eval"]["norm_amp"] = 1
    config["eval"]["background_amp"] = 0
    config["eval"]["measure_state"] = measure_state

    config["readout"]["relax_delay"] = 1500

    config["res"]["freq"] = config["res"]["freqs"][measure_state]

    disp1 = 13
    disp2 = 6
    disp_2_delay = 0

    # preparation_pulses = [
    #     pe.displacement(cavity="storage_A", alpha=disp1),
    #     pe.pi_pulse(rabi_states="ge", selective=True, gain_mult=2, meas_state=measure_state,
    #                 freq=config_dict["qubit"]["freqs"]["g"]["ge"] + (0 * config_dict["storage_A"]["chi"])),
    #     pe.displacement(cavity="storage_A", alpha=disp2, t=disp_2_delay, phase=180),
    # ]

    observable_dicts = [
        # [
        #     pe.pi_pulse(rabi_states="ef", meas_state=measure_state),
        #     pe.pi_pulse(rabi_states="ge", meas_state=measure_state),
        #     pe.pi_pulse(rabi_states="ge", meas_state=measure_state, selective=True,
        #                 freq=config_dict["qubit"]["freqs"]["g"]["ge"] + (
        #                             1 * config_dict["storage_A"]["chi"])),
        # ],
        [
            pe.pi_pulse(rabi_states="ef", meas_state=measure_state),
            pe.pi_pulse(rabi_states="ge", meas_state=measure_state, selective=True,
                        freq=config["qubit"]["freqs"]["g"]["ge"] + (1 * config["storage_A"]["chi"])),
        ],
        [
            pe.pi_pulse(rabi_states="ef", meas_state=measure_state),
            pe.pi_pulse(rabi_states="ge", meas_state=measure_state, gain_mult=0, selective=True),
        ],
    ]

    print("cost func config[expt]:", config["expt"])

    normVolt = ArbHamiltonianEval(soc=soc, soccfg=soccfg, config=config, observable_dicts=observable_dicts)

    # calculate drive amps for all times that the tprocessor will run for
    spread_drive_amps = []
    for i in range(num_drives):
        spread_drive_amps.append(
            qick_spread_pulse_amps(soccfg, times[2 * i], comp_amp_funcs[2 * i], drive_chs[i]))
        spread_drive_amps.append(
            qick_spread_pulse_amps(soccfg, times[(2 * i) + 1], comp_amp_funcs[(2 * i) + 1], drive_chs[i]))

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
            normVolt.pulse_view(title="|1g> trial pulses", n=i)

    [[i_arr_m0, i_arr_m1], [q_arr_m0, q_arr_m1]] = np.array(
        normVolt.measure(progress=True, save=False, all_shots=True))[:, :, 0, :]

    # if verbose:
    #     print(
    #         f'Measurement 0: i: avg: {np.average(i_arr_m0)}, std:{np.std(i_arr_m0)}, q: {np.average(q_arr_m0)}, std:{np.std(q_arr_m0)}')
    #     print(
    #         f'Measurement 1: i: avg: {np.average(i_arr_m1)}, std:{np.std(i_arr_m1)}, q: {np.average(q_arr_m1)}, std:{np.std(q_arr_m1)}')

    if verbose:
        print("m0: avgi:", np.average(i_arr_m0), "avgq:", np.average(q_arr_m0))
        print("m1: avgi:", np.average(i_arr_m1), "avgq:", np.average(q_arr_m1))

    mapped_probs_m0 = SSR_map(i_arr_m0, q_arr_m0, config["qubit"]["g_blob"], config["qubit"]["e_blob"])
    mapped_probs_m1 = SSR_map(i_arr_m1, q_arr_m1, config["qubit"]["g_blob"], config["qubit"]["e_blob"])

    avg_mapped_probs_m0 = np.average(mapped_probs_m0)
    avg_mapped_probs_m1 = np.average(mapped_probs_m1)

    std_mapped_probs_m0 = np.std(mapped_probs_m0)
    std_mapped_probs_m1 = np.std(mapped_probs_m1)

    if verbose:
        print(f'Measurement 0: avg: {avg_mapped_probs_m0}, std:{std_mapped_probs_m0}')
        print(f'Measurement 1: avg: {avg_mapped_probs_m1}, std:{std_mapped_probs_m1}')

    reward = avg_mapped_probs_m0 - avg_mapped_probs_m1

    # cost = round(np.abs(q_amp / contrast))  # + np.abs(np.sum(np.sqrt(np.power(i_amps, 2) + np.power(q_amps, 2))) / 100000000)
    if verbose:
        print(f'Previous reward: {reward}')

    return reward
