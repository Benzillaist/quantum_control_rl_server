"""
All in one calibration file
"""

import numpy as np
from utils import pulse_configs as pc
from utils.hamiltonians import ArbHamiltonianEval
import matplotlib.pyplot as plt
from utils.opt_utils import SSR_map, rabi_fit
import utils.pulse_elements as pe
import lmfit
from tqdm import tqdm
import json

# ==============================================
# Configuration dictionary

# import self.config
with open("../../utils/pulse_configs.py") as f:
    data = f.read()

# self.config = json.loads(data)

class Measurements:

    def __init__(self, soc, soccfg, config_dict, hdf5_file_name):

        self.soc = soc
        self.soccfg = soccfg
        self.config = config_dict
        self.hdf5_file_name = hdf5_file_name

    def ge_g_contrast(self, run, debug):
        expt_dict = {
            "reps": 100000,  # inner repeat times
            "rounds": 1,  # outer loop times
            "soft_avg": 1,  # soft average (only for decimated readout)
            "expts": 1,  # parameters sweep
            "hdf5": self.hdf5_file_name,
        }
        self.config["expt"] = expt_dict

        measure_state = "g"

        # Modify eval_dict to work for this experiment
        self.config["eval"]["coeffs"] = [1, -1]
        self.config["eval"]["norm_amp"] = 1
        self.config["eval"]["background_amp"] = 0
        self.config["eval"]["measure_state"] = measure_state

        self.config["res"]["freq"] = self.config["res"]["freqs"][measure_state]

        observable_dicts = [
            [
                pe.pi_pulse(rabi_states="ef", meas_state=measure_state),
                pe.pi_pulse(rabi_states="ge", meas_state=measure_state, selective=True),
            ],
            [
                pe.pi_pulse(rabi_states="ef", meas_state=measure_state),
            ],
        ]

        normVolt = ArbHamiltonianEval(soc=self.soc, soccfg=self.soccfg, config=self.config, observable_dicts=observable_dicts)

        if debug:
            for i in range(len(observable_dicts)):
                normVolt.pulse_view(title="g GE contrast", n=i)

        if run:
            contrast = normVolt.expect(progress=True, self_project=True)
            print(f'GF @e contrast: {contrast}')
            return contrast

    def ef_g_contrast(self, run, debug):
        expt_dict = {
            "reps": 100000,  # inner repeat times
            "rounds": 1,  # outer loop times
            "soft_avg": 1,  # soft average (only for decimated readout)
            "expts": 1,  # parameters sweep
            "hdf5": self.hdf5_file_name,
        }
        self.config["expt"] = expt_dict

        measure_state = "g"

        # Modify eval_dict to work for this experiment
        self.config["eval"]["coeffs"] = [1, -1]
        self.config["eval"]["norm_amp"] = 1
        self.config["eval"]["background_amp"] = 0
        self.config["eval"]["measure_state"] = measure_state

        self.config["res"]["freq"] = self.config["res"]["freqs"][measure_state]

        observable_dicts = [
            [
                pe.pi_pulse(rabi_states="ef", meas_state=measure_state),
            ],
            [
                None
            ],
        ]

        normVolt = ArbHamiltonianEval(soc=self.soc, soccfg=self.soccfg, config=self.config, observable_dicts=observable_dicts)

        if debug:
            for i in range(len(observable_dicts)):
                normVolt.pulse_view(title="e EF contrast", n=i)

        if run:
            contrast = normVolt.expect(progress=True, self_project=True)
            print(f'EF @e contrast: {contrast}')
            return contrast


    def ef_e_contrast(self, run, debug):
        expt_dict = {
            "reps": 1000000,  # inner repeat times
            "rounds": 1,  # outer loop times
            "soft_avg": 1,  # soft average (only for decimated readout)
            "expts": 1,  # parameters sweep
            "hdf5": self.hdf5_file_name,
        }
        self.config["expt"] = expt_dict

        measure_state = "e"

        # Modify eval_dict to work for this experiment
        self.config["eval"]["coeffs"] = [1, -1]
        self.config["eval"]["norm_amp"] = 1
        self.config["eval"]["background_amp"] = 0
        self.config["eval"]["measure_state"] = measure_state

        self.config["res"]["freq"] = self.config["res"]["freqs"][measure_state]

        observable_dicts = [
            [
                pe.pi_pulse(rabi_states="ef", meas_state=measure_state),
            ],
            [
                None
            ],
        ]

        normVolt = ArbHamiltonianEval(soc=self.soc, soccfg=self.soccfg, config=self.config, observable_dicts=observable_dicts)

        if debug:
            for i in range(len(observable_dicts)):
                normVolt.pulse_view(title="e EF contrast", n=i)

        if run:
            contrast = normVolt.expect(progress=True, self_project=True)
            print(f'EF @e contrast: {contrast}')
            return contrast


    def gf_e_contrast(self, run, debug):
        expt_dict = {
            "reps": 1000000,  # inner repeat times
            "rounds": 1,  # outer loop times
            "soft_avg": 1,  # soft average (only for decimated readout)
            "expts": 1,  # parameters sweep
            "hdf5": self.hdf5_file_name,
        }
        self.config["expt"] = expt_dict

        measure_state = "e"

        # Modify eval_dict to work for this experiment
        self.config["eval"]["coeffs"] = [1, -1]
        self.config["eval"]["norm_amp"] = 1
        self.config["eval"]["background_amp"] = 0
        self.config["eval"]["measure_state"] = measure_state

        self.config["res"]["freq"] = self.config["res"]["freqs"][measure_state]

        observable_dicts = [
            [
                pe.pi_pulse(rabi_states="ef", meas_state=measure_state),
                pe.pi_pulse(rabi_states="ef", meas_state=measure_state),
            ],
            [
                pe.pi_pulse(rabi_states="ge", meas_state=measure_state)
            ],
        ]

        normVolt = ArbHamiltonianEval(soc=self.soc, soccfg=self.soccfg, config=self.config, observable_dicts=observable_dicts)

        if debug:
            for i in range(len(observable_dicts)):
                normVolt.pulse_view(title="e GF contrast", n=i)

        if run:
            contrast = normVolt.expect(progress=True, self_project=True)
            print(f'GF @e contrast: {contrast}')
            return contrast

    def ge_projection_angle(self, run, debug):
        expt_dict = {
            "reps": 1000000,  # inner repeat times
            "rounds": 1,  # outer loop times
            "soft_avg": 1,  # soft average (only for decimated readout)
            "expts": 1,  # parameters sweep
            "hdf5": self.hdf5_file_name,
        }
        self.config["expt"] = expt_dict

        measure_state = "g"

        # Modify eval_dict to work for this experiment
        self.config["eval"]["coeffs"] = [1, -1]
        self.config["eval"]["norm_amp"] = 1
        self.config["eval"]["background_amp"] = 0
        self.config["eval"]["measure_state"] = measure_state

        self.config["res"]["freq"] = self.config["res"]["freqs"][measure_state]

        observable_dicts = [
            [
                None
            ],
            [
                pe.pi_pulse(rabi_states="ge", meas_state=measure_state),
            ],
        ]

        normVolt = ArbHamiltonianEval(soc=self.soc, soccfg=self.soccfg, config=self.config, observable_dicts=observable_dicts)

        if debug:
            for i in range(len(observable_dicts)):
                normVolt.pulse_view(title="GE projection angle", n=i)

        if run:
            avgi_arr, avgq_arr = normVolt.measure(progress=True)
            angle = -np.arctan2((avgq_arr[1] - avgq_arr[0]), (avgi_arr[1] - avgi_arr[0]))
            print(f'GE angle: {angle}')
            return angle

    def ef_projection_angle(self, run, debug):
        expt_dict = {
            "reps": 1000000,  # inner repeat times
            "rounds": 1,  # outer loop times
            "soft_avg": 1,  # soft average (only for decimated readout)
            "expts": 1,  # parameters sweep
            "hdf5": self.hdf5_file_name,
        }
        self.config["expt"] = expt_dict

        measure_state = "e"

        # Modify eval_dict to work for this experiment
        self.config["eval"]["coeffs"] = [1, -1]
        self.config["eval"]["norm_amp"] = 1
        self.config["eval"]["background_amp"] = 0
        self.config["eval"]["measure_state"] = measure_state

        self.config["res"]["freq"] = self.config["res"]["freqs"][measure_state]

        observable_dicts = [
            [
                pe.pi_pulse(rabi_states="ef", meas_state=measure_state),
            ],
            [
                pe.pi_pulse(rabi_states="ge", meas_state=measure_state),
            ],
        ]

        normVolt = ArbHamiltonianEval(soc=self.soc, soccfg=self.soccfg, config=self.config, observable_dicts=observable_dicts)

        if debug:
            for i in range(len(observable_dicts)):
                normVolt.pulse_view(title="EF projection angle", n=i)

        if run:
            avgi_arr, avgq_arr = normVolt.measure(progress=True)
            angle = -np.arctan2((avgq_arr[1] - avgq_arr[0]), (avgi_arr[1] - avgi_arr[0]))
            print(f'EF angle: {angle}')
            return angle


    def gf_projection_angle(self, run, debug):
        expt_dict = {
            "reps": 1000000,  # inner repeat times
            "rounds": 1,  # outer loop times
            "soft_avg": 1,  # soft average (only for decimated readout)
            "expts": 1,  # parameters sweep
            "hdf5": self.hdf5_file_name,
        }
        self.config["expt"] = expt_dict

        measure_state = "g"

        # Modify eval_dict to work for this experiment
        self.config["eval"]["coeffs"] = [1, -1]
        self.config["eval"]["norm_amp"] = 1
        self.config["eval"]["background_amp"] = 0
        self.config["eval"]["measure_state"] = measure_state

        self.config["res"]["freq"] = self.config["res"]["freqs"][measure_state]

        observable_dicts = [
            [

            ],
            [
                pe.pi_pulse(rabi_states="ge", meas_state=measure_state),
                pe.pi_pulse(rabi_states="ef", meas_state=measure_state),
            ],
        ]

        normVolt = ArbHamiltonianEval(soc=self.soc, soccfg=self.soccfg, config=self.config, observable_dicts=observable_dicts)

        if debug:
            for i in range(len(observable_dicts)):
                normVolt.pulse_view(title="GF projection angle", n=i)

        if run:
            avgi_arr, avgq_arr = normVolt.measure(progress=True)
            angle = -np.arctan2((avgq_arr[1] - avgq_arr[0]), (avgi_arr[1] - avgi_arr[0]))
            print(f'GF angle: {angle}')
            return angle

    def c0g_contrast(self, run, debug):
        expt_dict = {
            "reps": 100000,  # inner repeat times
            "rounds": 1,  # outer loop times
            "soft_avg": 1,  # soft average (only for decimated readout)
            "expts": 1,  # parameters sweep
            "hdf5": self.hdf5_file_name,
        }
        self.config["expt"] = expt_dict

        measure_state = "g"
        storage = "A"

        # Modify eval_dict to work for this experiment
        self.config["eval"]["coeffs"] = [1, -1]
        self.config["eval"]["norm_amp"] = 1
        self.config["eval"]["background_amp"] = 0
        self.config["eval"]["measure_state"] = measure_state

        self.config["res"]["freq"] = self.config["res"]["freqs"][measure_state]

        disp_alpha = 1

        observable_dicts = [
            [
                # pe.displacement(cavity="storage_" + storage, alpha=disp_alpha),
                pe.pi_pulse(rabi_states="ef", meas_state=measure_state),
            ],
            [
                # pe.displacement(cavity="storage_" + storage, alpha=disp_alpha),
                pe.pi_pulse(rabi_states="ef", meas_state=measure_state),
                pe.pi_pulse(rabi_states="ge", meas_state=measure_state, selective=True),
            ],
        ]

        normVolt = ArbHamiltonianEval(soc=self.soc, soccfg=self.soccfg, config=self.config, observable_dicts=observable_dicts)

        if debug:
            for i in range(len(observable_dicts)):
                normVolt.pulse_view(title="|0g> contrast", n=i)

        if run:
            contrast = abs(normVolt.expect(progress=True, self_project=True, save=False))
            print(f'0g_contrast: {contrast}')
            return contrast

    def c1g_contrast(self, run, debug, state_prep = None):
        expt_dict = {
            "reps": 100000,  # inner repeat times
            "rounds": 1,  # outer loop times
            "soft_avg": 1,  # soft average (only for decimated readout)
            "expts": 1,  # parameters sweep
            "hdf5": self.hdf5_file_name,
        }
        self.config["expt"] = expt_dict

        measure_state = "g"
        storage = "B"

        # Modify eval_dict to work for this experiment
        self.config["eval"]["coeffs"] = [1, -1]
        self.config["eval"]["norm_amp"] = 1
        self.config["eval"]["background_amp"] = 0
        self.config["eval"]["measure_state"] = measure_state

        self.config["res"]["freq"] = self.config["res"]["freqs"][measure_state]

        disp_alpha_1 = 1
        disp_alpha_2 = 0.7

        observable_dicts = [
            [
                pe.pi_pulse(rabi_states="ef", meas_state=measure_state),
            ],
            [
                pe.pi_pulse(rabi_states="ef", meas_state=measure_state),
                pe.pi_pulse(rabi_states="ge", meas_state=measure_state, selective=True,
                            freq=self.config["qubit"]["freqs"]["g"]["ge"] + self.config["storage_" + storage][
                                "chi"]),
            ],
        ]

        normVolt = ArbHamiltonianEval(soc=self.soc, soccfg=self.soccfg, config=self.config, observable_dicts=observable_dicts)

        # normVolt.add_test_waveform(self.config["storage_" + storage]["ch"], {
        #     "name": f'test_wf',
        #     "I": np.ones(1376) * 2000,
        #     "Q": np.zeros(1376),
        #     "gain": 32622,
        #     "freq": self.config["storage_" + storage]["freq"],
        #     "phase": 0,
        #     "t": 'auto' # soccfg.us2cycles(gen_ch=ch, us=0.05) + 16,
        # }),

        # storage_ch = self.config["storage_A"]["ch"]
        # storage_sigma = 0.01  # [us]
        #
        # normVolt.add_test_waveform(storage_ch, {
        #     "name": "displace_0",  # Name of pulse, should be unique across program
        #     "I": normVolt.gauss_amps(ch=storage_ch, sigma=soccfg.us2cycles(gen_ch=storage_ch, us=storage_sigma),
        #                              length=soccfg.us2cycles(gen_ch=storage_ch, us=storage_sigma) * 4),
        #     # I pulse values (0 <= I < 32768)
        #     "Q": np.zeros(16 * soccfg.us2cycles(gen_ch=storage_ch, us=storage_sigma) * 4),
        #     # Q pulse values (0 <= I < 32768)
        #     "gain": int(self.config["storage_A"]["alpha_amp"] * 1.14),  # Gain of pulse
        #     "freq": self.config["storage_A"]["freq"],  # Frequency to play pulse at
        #     "phase": 0,  # Phase of pulse
        #     "t": 'auto'  # Time at which pulse is sent
        # }, )
        #
        # qubit_ch = self.config["qubit"]["ch"]
        # qubit_sigma = self.config["qubit"]["ge"]["selective_sigma"]
        #
        # normVolt.add_test_waveform(qubit_ch, {
        #     "name": "selective_ge_pi",
        #     "I": normVolt.gauss_amps(ch=qubit_ch, sigma=soccfg.us2cycles(gen_ch=qubit_ch, us=qubit_sigma),
        #                              length=soccfg.us2cycles(gen_ch=qubit_ch, us=qubit_sigma) * 4),
        #     "Q": np.zeros(16 * soccfg.us2cycles(gen_ch=qubit_ch, us=qubit_sigma) * 4),
        #     "gain": self.config["qubit"]["ge"]["selective_pi_gain"] * 2,
        #     "freq": self.config["qubit"]["freqs"]["g"]["ge"],
        #     "phase": 0,
        #     "t": soccfg.us2cycles(gen_ch=qubit_ch, us=0.05) + 16,
        # }, )
        #
        # normVolt.add_test_waveform(storage_ch, {
        #     "name": "displace_1",
        #     "I": normVolt.gauss_amps(ch=storage_ch, sigma=soccfg.us2cycles(gen_ch=storage_ch, us=storage_sigma),
        #                              length=soccfg.us2cycles(gen_ch=storage_ch, us=storage_sigma) * 4),
        #     "Q": np.zeros(16 * soccfg.us2cycles(gen_ch=storage_ch, us=storage_sigma) * 4),
        #     "gain": int(self.config["storage_A"]["alpha_amp"] * 0.57),
        #     "freq": self.config["storage_A"]["freq"],
        #     "phase": 180,
        #     "t": (soccfg.us2cycles(gen_ch=qubit_ch, us=qubit_sigma + 0.0125) * 4) + 24,
        # }, )

        if debug:
            for i in range(len(observable_dicts)):
                normVolt.pulse_view(title="|1g> contrast", n=i)

        if run:
            contrast = abs(normVolt.expect(progress=True, self_project=True, save=False))
            print(f'1g_contrast: {contrast}')
            return contrast

    def c2g_contrast(self, run, debug, state_prep = None):
        expt_dict = {
            "reps": 100000,  # inner repeat times
            "rounds": 1,  # outer loop times
            "soft_avg": 1,  # soft average (only for decimated readout)
            "expts": 1,  # parameters sweep
            "hdf5": self.hdf5_file_name,
        }
        self.config["expt"] = expt_dict

        measure_state = "g"
        storage = "A"

        # Modify eval_dict to work for this experiment
        self.config["eval"]["coeffs"] = [1, -1]
        self.config["eval"]["norm_amp"] = 1
        self.config["eval"]["background_amp"] = 0
        self.config["eval"]["measure_state"] = measure_state

        self.config["res"]["freq"] = self.config["res"]["freqs"][measure_state]

        disp_alpha = 1

        observable_dicts = [
            [
                pe.displacement(cavity="storage_A", alpha=disp_alpha),
                pe.pi_pulse(rabi_states="ef", meas_state=measure_state),
            ],
            [
                pe.displacement(cavity="storage_A", alpha=disp_alpha),
                pe.pi_pulse(rabi_states="ef", meas_state=measure_state),
                pe.pi_pulse(rabi_states="ge", meas_state=measure_state, selective=True,
                            freq=self.config["qubit"]["freqs"]["g"]["ge"] + (
                                        2 * self.config["storage_" + storage]["chi"])),
            ],
        ]

        normVolt = ArbHamiltonianEval(soc=self.soc, soccfg=self.soccfg, config=self.config, observable_dicts=observable_dicts)

        if debug:
            for i in range(len(observable_dicts)):
                normVolt.pulse_view(title="|0g> contrast", n=i)

        if run:
            contrast = abs(normVolt.expect(progress=True, self_project=False, save=False))
            print(f'2gf_contrast: {contrast}')
            return contrast

    def single_shot(self, run, debug, state_prep = None, reps = 100000):
        expt_dict = {
            "reps": reps,  # inner repeat times
            "rounds": 1,  # outer loop times
            "soft_avg": 1,  # soft average (only for decimated readout)
            "expts": 1,  # parameters sweep
            "hdf5": self.hdf5_file_name,
        }
        self.config["expt"] = expt_dict

        measure_state = "g"

        # Modify eval_dict to work for this experiment
        self.config["eval"]["coeffs"] = [1, -1]
        self.config["eval"]["norm_amp"] = 1
        self.config["eval"]["background_amp"] = 0
        self.config["eval"]["measure_state"] = measure_state
        self.config["readout"]["relax_delay"] = 300

        self.config["res"]["freq"] = self.config["res"]["freqs"][measure_state]

        observable_dicts = [
            [
            ],
            [
                pe.pi_pulse(rabi_states="ge", meas_state=measure_state),
            ],
            [
                pe.pi_pulse(rabi_states="ge", meas_state=measure_state),
                pe.pi_pulse(rabi_states="ef", meas_state=measure_state),
                pe.pi_pulse(rabi_states="ge", meas_state=measure_state),
            ],
        ]

        normVolt = ArbHamiltonianEval(soc=self.soc, soccfg=self.soccfg, config=self.config,
                                      observable_dicts=observable_dicts)

        if debug:
            for i in range(len(observable_dicts)):
                normVolt.pulse_view(title="|0g> contrast", n=i)

        if run:
            [[gi, ei, fi], [gq, eq, fq]] = np.array(normVolt.measure(progress=True, save=False, all_shots=True))[:, :, 0, :]


            # print(f'g_blob: {g_blob}')
            # print(f'e_blob: {e_blob}')
            # print(f'f_blob: {f_blob}')

            print(f'g_blob: {np.average(gi)}, {np.average(gq)}')
            print(f'e_blob: {np.average(ei)}, {np.average(eq)}')
            print(f'f_blob: {np.average(fi)}, {np.average(fq)}')

            plt.plot(gi, gq, label="|g>", linestyle="None", marker="*", alpha=0.1)
            plt.plot(ei, eq, label="|e>", linestyle="None", marker="*", alpha=0.1)
            plt.plot(fi, fq, label="|f>", linestyle="None", marker="*", alpha=0.1)
            plt.legend()
            plt.show()

            return np.array([[np.average(gi), np.average(gq)], [np.average(ei), np.average(eq)], [np.average(fi), np.average(fq)]])

    def SNAP_calibration(self, run, debug, state_prep = None, reps = 3000):
        expt_dict = {
            "reps": reps,  # inner repeat times
            "rounds": 1,  # outer loop times
            "soft_avg": 1,  # soft average (only for decimated readout)
            "expts": 1,  # parameters sweep
            "hdf5": self.hdf5_file_name,
        }
        self.config["expt"] = expt_dict

        measure_state = "g"

        # Modify eval_dict to work for this experiment
        self.config["eval"]["coeffs"] = [1, -1]
        self.config["eval"]["norm_amp"] = 1
        self.config["eval"]["background_amp"] = 0
        self.config["eval"]["measure_state"] = measure_state
        self.config["readout"]["relax_delay"] = 2500

        self.config["res"]["freq"] = self.config["res"]["freqs"][measure_state]

        pop_arr = []

        min_disp1 = 8  # [us]
        max_disp1 = 16  # [us]
        12, 6
        6, 3
        8+6, 4+3
        14, 7
        min_disp2 = 4  # [us]
        max_disp2 = 12  # [us]

        disp_2_delay = 0  # [tproc cycles]

        disp_arr_1 = np.linspace(min_disp1, max_disp1, 17)
        disp_arr_2 = np.linspace(min_disp2, max_disp2, 17)

        for i, disp1 in enumerate(disp_arr_1):
            pop_arr_temp = []
            for j, disp2 in enumerate(disp_arr_2):

                print(f'i: {i}, j: {j}')
                print(f'disp1: {disp1}, disp2: {disp2}')

                preparation_pulses = [
                    pe.displacement(cavity="storage_A", alpha=disp1),
                    pe.pi_pulse(rabi_states="ge", selective=True, gain_mult=2, meas_state=measure_state,
                                freq=self.config["qubit"]["freqs"]["g"]["ge"] + (0 * self.config["storage_A"]["chi"])),
                    pe.displacement(cavity="storage_A", alpha=disp2, t=disp_2_delay, phase=180),
                ]

                observable_dicts = [
                    # [
                    #     pe.pi_pulse(rabi_states="ef", meas_state=measure_state),
                    #     pe.pi_pulse(rabi_states="ge", meas_state=measure_state),
                    #     pe.pi_pulse(rabi_states="ge", meas_state=measure_state, selective=True,
                    #                 freq=self.config["qubit"]["freqs"]["g"]["ge"] + (1 * self.config["storage_A"]["chi"])),
                    # ],
                    [
                        pe.pi_pulse(rabi_states="ef", meas_state=measure_state),
                        pe.pi_pulse(rabi_states="ge", meas_state=measure_state, selective=True,
                                    freq=self.config["qubit"]["freqs"]["g"]["ge"] + (
                                            1 * self.config["storage_A"]["chi"])),
                    ],
                    [
                        pe.pi_pulse(rabi_states="ef", meas_state=measure_state),
                        pe.pi_pulse(rabi_states="ge", meas_state=measure_state, gain_mult=0, selective=True),
                    ],
                ]

                normVolt = ArbHamiltonianEval(soc=self.soc, soccfg=self.soccfg, config=self.config,
                                              preparation_pulses=preparation_pulses, observable_dicts=observable_dicts)

                if debug:
                    for i in range(len(observable_dicts)):
                        normVolt.pulse_view(title="|0g> contrast", n=i)

                if run:
                    [[i_arr_m0, i_arr_m1], [q_arr_m0, q_arr_m1]] = np.array(normVolt.measure(progress=True, save=False, all_shots=True))[:, :, 0, :]

                    mapped_probs_m0 = SSR_map(i_arr_m0, q_arr_m0, self.config["qubit"]["g_blob"], self.config["qubit"]["e_blob"])
                    mapped_probs_m1 = SSR_map(i_arr_m1, q_arr_m1, self.config["qubit"]["g_blob"], self.config["qubit"]["e_blob"])

                    avg_mapped_probs_m0 = np.average(mapped_probs_m0)
                    avg_mapped_probs_m1 = np.average(mapped_probs_m1)

                    reward = avg_mapped_probs_m0 - avg_mapped_probs_m1

                    # mapped_vals = SSR_map(i_arr, q_arr, self.config["qubit"]["g_blob"], self.config["qubit"]["e_blob"])

                    pop_arr_temp.append(reward)

            pop_arr.append(pop_arr_temp)

        pop_arr = np.array(pop_arr)

        # print(f'Max amp: {disp_arr[np.argmax(pop_arr)]}')
        # print(f'Max prob: {max(pop_arr)}')

        # print()

        plt.imshow(pop_arr)
        plt.xlabel("Displacement 2 (kDAC units)")
        plt.ylabel("Displacement 1 (kDAC units)")

        plt.colorbar()

        plt.show()

        return disp_arr_1, disp_arr_2, pop_arr

    def state_characterization(self, run, debug, state_prep=None, reps = 10000):
        expt_dict = {
            "reps": reps,  # inner repeat times
            "rounds": 1,  # outer loop times
            "soft_avg": 1,  # soft average (only for decimated readout)
            "expts": 1,  # parameters sweep
            "hdf5": self.hdf5_file_name,
        }
        self.config["expt"] = expt_dict

        measure_state = "g"

        # Modify eval_dict to work for this experiment
        self.config["eval"]["coeffs"] = [1, -1]
        self.config["eval"]["norm_amp"] = 1
        self.config["eval"]["background_amp"] = 0
        self.config["eval"]["measure_state"] = measure_state
        self.config["readout"]["relax_delay"] = 2000

        self.config["res"]["freq"] = self.config["res"]["freqs"][measure_state]

        disp1 = 11
        disp2 = 6
        disp_2_delay = 0

        preparation_pulses = [
            pe.displacement(cavity="storage_A", alpha=disp1),
            pe.pi_pulse(rabi_states="ge", selective=True, gain_mult=2, meas_state=measure_state),
            pe.displacement(cavity="storage_A", alpha=disp2, t=disp_2_delay, phase=180),
        ]

        observable_dicts = [
            # |0g>
            [
                pe.pi_pulse(rabi_states="ef", meas_state=measure_state),
                pe.pi_pulse(rabi_states="ge", meas_state=measure_state),
                pe.pi_pulse(rabi_states="ge", meas_state=measure_state, selective=True),

            ],
            # |0e>
            [
                pe.pi_pulse(rabi_states="ge", meas_state=measure_state),
                pe.pi_pulse(rabi_states="ef", meas_state=measure_state),
                pe.pi_pulse(rabi_states="ge", meas_state=measure_state),
                pe.pi_pulse(rabi_states="ge", meas_state=measure_state, selective=True),
            ],
            # |1g>
            [
                pe.pi_pulse(rabi_states="ef", meas_state=measure_state),
                pe.pi_pulse(rabi_states="ge", meas_state=measure_state),
                pe.pi_pulse(rabi_states="ge", meas_state=measure_state, selective=True,
                            freq=self.config["qubit"]["freqs"]["g"]["ge"] + (1 * self.config["storage_A"]["chi"])),
            ],
            # |1e>
            [
                pe.pi_pulse(rabi_states="ge", meas_state=measure_state),
                pe.pi_pulse(rabi_states="ef", meas_state=measure_state),
                pe.pi_pulse(rabi_states="ge", meas_state=measure_state),
                pe.pi_pulse(rabi_states="ge", meas_state=measure_state, selective=True,
                            freq=self.config["qubit"]["freqs"]["g"]["ge"] + (1 * self.config["storage_A"]["chi"])),
            ],
            # |2g>
            [
                pe.pi_pulse(rabi_states="ef", meas_state=measure_state),
                pe.pi_pulse(rabi_states="ge", meas_state=measure_state),
                pe.pi_pulse(rabi_states="ge", meas_state=measure_state, selective=True,
                            freq=self.config["qubit"]["freqs"]["g"]["ge"] + (
                                        2 * self.config["storage_A"]["chi"])),
            ],
            # |2e>
            [
                pe.pi_pulse(rabi_states="ge", meas_state=measure_state),
                pe.pi_pulse(rabi_states="ef", meas_state=measure_state),
                pe.pi_pulse(rabi_states="ge", meas_state=measure_state),
                pe.pi_pulse(rabi_states="ge", meas_state=measure_state, selective=True,
                            freq=self.config["qubit"]["freqs"]["g"]["ge"] + (
                                        2 * self.config["storage_A"]["chi"])),
            ],
            # |3g>
            [
                pe.pi_pulse(rabi_states="ef", meas_state=measure_state),
                pe.pi_pulse(rabi_states="ge", meas_state=measure_state),
                pe.pi_pulse(rabi_states="ge", meas_state=measure_state, selective=True,
                            freq=self.config["qubit"]["freqs"]["g"]["ge"] + (
                                    3 * self.config["storage_A"]["chi"])),
            ],
            # |3e>
            [
                pe.pi_pulse(rabi_states="ge", meas_state=measure_state),
                pe.pi_pulse(rabi_states="ef", meas_state=measure_state),
                pe.pi_pulse(rabi_states="ge", meas_state=measure_state),
                pe.pi_pulse(rabi_states="ge", meas_state=measure_state, selective=True,
                            freq=self.config["qubit"]["freqs"]["g"]["ge"] + (
                                    3 * self.config["storage_A"]["chi"])),
            ],
        ]

        normVolt = ArbHamiltonianEval(soc=self.soc, soccfg=self.soccfg, config=self.config,
                                      preparation_pulses=preparation_pulses, observable_dicts=observable_dicts)

        if debug:
            for i in range(len(observable_dicts)):
                normVolt.pulse_view(title="|0g> contrast", n=i)

        if run:
            [i_arr, q_arr] = np.array(normVolt.measure(progress=True, save=False, all_shots=True))[:, :, 0, :]

            i_arr = np.array(i_arr)
            q_arr = np.array(q_arr)

            state_pops = []

            for i in range(len(i_arr)):

                mapped_vals = SSR_map(i_arr[i], q_arr[i], self.config["qubit"]["g_blob"], self.config["qubit"]["e_blob"])

                state_pops.append(np.average(mapped_vals))

            state_pops = np.array(state_pops)

            state_pops = np.reshape(state_pops, (int(len(state_pops) / 2), 2))

            print(f'State populations:')
            print(f'{state_pops}')

            return state_pops

        return 0

    def ge_power_rabi(self, run, debug, state_prep=None, reps = 5000):
        expt_dict = {
            "reps": reps,  # inner repeat times
            "rounds": 1,  # outer loop times
            "soft_avg": 1,  # soft average (only for decimated readout)
            "expts": 1,  # parameters sweep
            "hdf5": self.hdf5_file_name,
        }
        self.config["expt"] = expt_dict

        measure_state = "g"

        # Modify eval_dict to work for this experiment
        self.config["eval"]["coeffs"] = [1, -1]
        self.config["eval"]["norm_amp"] = 1
        self.config["eval"]["background_amp"] = 0
        self.config["eval"]["measure_state"] = measure_state
        self.config["readout"]["relax_delay"] = 300

        self.config["res"]["freq"] = self.config["res"]["freqs"][measure_state]

        start_amp = 1000
        stop_amp = 30000
        steps = 101

        amps = np.linspace(start_amp, stop_amp, steps).astype(int)

        pop_arr = []

        for i, amp in enumerate(tqdm(amps)):
            preparation_pulses = [
                pe.pi_pulse(rabi_states="ge", gain=amp, meas_state=measure_state),
            ]

            observable_dicts = [
                [
                ],
            ]

            normVolt = ArbHamiltonianEval(soc=self.soc, soccfg=self.soccfg, config=self.config,
                                          preparation_pulses=preparation_pulses, observable_dicts=observable_dicts)

            if debug:
                for i in range(len(observable_dicts)):
                    normVolt.pulse_view(title="GE power rabi", n=i)

            if run:
                [[i_arr], [q_arr]] = np.array(normVolt.measure(progress=False, save=False, all_shots=True))[:,
                                     :, 0, :]

                mapped_vals = SSR_map(i_arr, q_arr, self.config["qubit"]["g_blob"], self.config["qubit"]["e_blob"])

                pop_arr.append(np.average(mapped_vals))

        x_pts = amps
        y_pts = pop_arr

        if y_pts[np.argmin(np.abs(x_pts))] < np.average(y_pts):
            amp0 = -(np.max(y_pts) - np.min(y_pts)) / 2
        else:
            amp0 = (np.max(y_pts) - np.min(y_pts)) / 2

        fftys = np.abs(np.fft.fft(y_pts - np.average(y_pts)))
        fftfs = np.fft.fftfreq(len(y_pts), x_pts[1] - x_pts[0])
        period0 = 1 / np.abs(fftfs[np.argmax(fftys)])
        params = lmfit.Parameters()
        params.add('ofs', value=np.average(y_pts))
        params.add('amp', value=amp0)
        params.add('period', value=period0, min=0)

        result = lmfit.minimize(rabi_fit, params, args=(x_pts, y_pts))
        lmfit.report_fit(result.params)

        plt.plot(x_pts, y_pts, "-o", markersize=4)
        plt.plot(x_pts, -rabi_fit(result.params, x_pts, 0),
                 label='Fit, pi_amp=%.01f, contrast=%.04f' % (
                     abs(result.params['period'].value / 2),
                     result.params['amp'].value * 2))
        plt.legend()
        plt.xlabel('amplitude')
        plt.title('GE Power Rabi')

        plt.show()

    def ef_power_rabi(self, run, debug, state_prep=None, reps = 5000):
        expt_dict = {
            "reps": reps,  # inner repeat times
            "rounds": 1,  # outer loop times
            "soft_avg": 1,  # soft average (only for decimated readout)
            "expts": 1,  # parameters sweep
            "hdf5": self.hdf5_file_name,
        }
        self.config["expt"] = expt_dict

        measure_state = "g"

        # Modify eval_dict to work for this experiment
        self.config["eval"]["coeffs"] = [1, -1]
        self.config["eval"]["norm_amp"] = 1
        self.config["eval"]["background_amp"] = 0
        self.config["eval"]["measure_state"] = measure_state
        self.config["readout"]["relax_delay"] = 300

        self.config["res"]["freq"] = self.config["res"]["freqs"][measure_state]

        start_amp = 1000
        stop_amp = 30000
        steps = 101

        amps = np.linspace(start_amp, stop_amp, steps).astype(int)

        pop_arr = []

        for i, amp in enumerate(tqdm(amps)):
            preparation_pulses = [
                pe.pi_pulse(rabi_states="ge", gain=amp, meas_state=measure_state),
                pe.pi_pulse(rabi_states="ef", gain=amp, meas_state=measure_state),
                pe.pi_pulse(rabi_states="ge", gain=amp, meas_state=measure_state),
            ]

            observable_dicts = [
                [
                ],
            ]

            normVolt = ArbHamiltonianEval(soc=self.soc, soccfg=self.soccfg, config=self.config,
                                          preparation_pulses=preparation_pulses, observable_dicts=observable_dicts)

            if debug:
                for i in range(len(observable_dicts)):
                    normVolt.pulse_view(title="|0g> contrast", n=i)

            if run:
                [[i_arr], [q_arr]] = np.array(normVolt.measure(progress=False, save=False, all_shots=True))[:,
                                     :, 0, :]

                mapped_vals = SSR_map(i_arr, q_arr, self.config["qubit"]["g_blob"], self.config["qubit"]["e_blob"])

                pop_arr.append(np.average(mapped_vals))

        x_pts = amps
        y_pts = pop_arr

        if y_pts[np.argmin(np.abs(x_pts))] < np.average(y_pts):
            amp0 = -(np.max(y_pts) - np.min(y_pts)) / 2
        else:
            amp0 = (np.max(y_pts) - np.min(y_pts)) / 2

        fftys = np.abs(np.fft.fft(y_pts - np.average(y_pts)))
        fftfs = np.fft.fftfreq(len(y_pts), x_pts[1] - x_pts[0])
        period0 = 1 / np.abs(fftfs[np.argmax(fftys)])
        params = lmfit.Parameters()
        params.add('ofs', value=np.average(y_pts))
        params.add('amp', value=amp0)
        params.add('period', value=period0, min=0)

        result = lmfit.minimize(rabi_fit, params, args=(x_pts, y_pts))
        lmfit.report_fit(result.params)

        plt.plot(x_pts, y_pts, "-o", markersize=4)
        plt.plot(x_pts, -rabi_fit(result.params, x_pts, 0),
                 label='Fit, pi_amp=%.01f, contrast=%.04f' % (
                     abs(result.params['period'].value / 2),
                     result.params['amp'].value * 2))
        plt.legend()
        plt.xlabel('amplitude')
        plt.title('EF Power Rabi')

        plt.show()
