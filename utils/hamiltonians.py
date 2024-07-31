from utils.arb_evaluate import ArbitraryEvaluate
import numpy as np
from utils import Operations, ConfigObj
from utils.pulse_sequences import PulseSequences


class ArbHamiltonianEval(Operations):
    def __init__(self, soc, soccfg, config: dict, observable_dicts=[], wavefunc_gen=None):
        """
        Initializes values for Hamiltonian object, calculates the wave function needed to evaluate the given
        observables for the Hamiltonian and wave function.

        As of right now, can only deal with photon counting operators and photon transfer operators (can actually deal with neither of these operators as of right now

        See test_arb_evaluate.py for an example of how the configuration dictionary should be formatted

        :param soc:
        :param soccfg:
        :param config:
        :param observable_dicts:
        :param wavefunc_gen:
        """
        """Initializes values for Hamiltonian object, calculates the wave function needed to evaluate the given 
        observables for the Hamiltonian and wave function. As of right now, can only deal with photon counting 
        operators and photon transfer operators

        :param coeffs: list of coefficient calculation functions, takes in either 1 or 2 wave function and returns scalar
        :param hamil_ops: operators corresponding to the coefficients, list of uniformly long lists of tuples
        :param wavefunc_ops: operators corresponding to the coefficients of , list of uniformly long lists of tuples
        :param wavefunc: parameterized generator ansatz of wave function. Single element that is used to compose larger, total wave function (elements are added on top of each other)s
        """

        # Holds information to be used later
        self.soc = soc
        self.soccfg = soccfg
        self.config = config
        # self.coeffs = coeffs
        self.observable_dicts = observable_dicts
        self.wavefunc_gen = wavefunc_gen
        self.observables = []

        # self.qubit_chs = []
        # self.cavity_chs = []

        self.channel_ids = {}

        # Add
        for ch_config in config.keys():
            if 'ch' in config[ch_config]:
                self.channel_ids[config[ch_config]['ch']] = ch_config

        # for dict_name in cavity_dict_names:
        #     # self.cavity_chs.append(config['ch'])
        #     self.channel_ids[config['ch']] = dict_name

        # # Process observables, add pulses in to config
        # for i, observable in enumerate(observable_dicts):
        #     # If it is a photon counting operator
        #     if observable[0] == observable[1]:
        #         self.observables.append({
        #             'ch': 0,
        #             'nqz': 2,
        #             'name': "observable" + str(i)
        #         })
        #         print("test1")
        #     # If it is a photon transfer operator
        #     elif observable[0] == observable[1][::-1]:
        #         print("test")
        #     else:
        #         raise Exception("Unrecognized operator, try again >:)")

    def measure(self, save=True, progress=False):
        """
        Returns a list of the Hamiltonian operators and wavefunction operators as previously defined
        For each wavefunction's I and Q data within the dictionary, each index relates to 0.145321801 ns worth of pulse time on the ZCU 216 QiCK board, length must be divisible by 16.
        The format of the pulse lists should be: [[I data], [Q data]]

        :return Voltage average of each measurement
        """

        # # Checks whether the I and Q data values have valid lengths
        # for i in range(len(qubit_pulses)):
        #     if len(qubit_pulses[i][0]) % 16 is not 0 or len(qubit_pulses[i][1]) % 16 is not 0:
        #         raise Exception("Length of I and Q data values should be divisible by 16")
        #
        # for i in range(len(cavity_pulses)):
        #     if len(cavity_pulses[i][0]) % 16 is not 0 or len(cavity_pulses[i][1]) % 16 is not 0:
        #         raise Exception("Length of I and Q data values should be divisible by 16")

        # Create empty results list
        avgi_list = []
        avgq_list = []

        # For each observable
        for observable_dict in self.observable_dicts:
            # For each observable to be measured
            self.config["observable_pulses"] = observable_dict

            # Create QiCK program object
            prog_obj = ArbitraryEvaluate(soc=self.soc, soccfg=self.soccfg, config=self.config)

            # Run QiCK program
            avgi, avgq = prog_obj.run(progress=progress)

            # Save averages for each observable by appending to list
            avgi_list.append(avgi)
            avgq_list.append(avgq)

        # Convert lists to numpy arrays
        avgi_arr = np.array(avgi_list)
        avgq_arr = np.array(avgq_list)

        # Save data
        if save:
            data_dict = {"avgi": avgi_arr, "avgq": avgq_arr}
            self.save_data(data=data_dict, name="arb_evaluate_measure", file_name=self.config["expt"]["hdf5"])

        print(avgi_arr)
        print(avgq_arr)

        return avgi_arr, avgq_arr

    def expect(self, save=True, progress=False, self_project=False):
        """
        Returns the expectation value of the Hamiltonian modeled by the observables

        :return:
        """
        eval_dict = self.config['eval']

        rot_res = 0

        # Gets I and Q values
        avgi_arr, avgq_arr = self.measure(save=save, progress=progress)

        if len(avgi_arr) == 2:
            [unrot_i0, unrot_i1], [unrot_q0, unrot_q1] = avgi_arr, avgq_arr

            # Finds projection rotation angle
            rot_angle = 0
            if self_project:
                rot_angle = -np.arctan2((unrot_q1 - unrot_q0), (unrot_i1 - unrot_i0))
                print(f'Self projection angle: {rot_angle}')
            else:
                rot_angle = self.config["eval"]["proj_rotation"][self.config["eval"]["measure_state"]]

            # print(f'Rotation angle: {rot_angle}')

            # Finds rotated I and Q values to use as projection basis
            rot_i0 = unrot_i0 * np.cos(rot_angle) - unrot_q0 * np.sin(rot_angle)
            # rot_q0 = unrot_i0 * np.sin(rot_angle) + unrot_q0 * np.cos(rot_angle)
            rot_i1 = unrot_i1 * np.cos(rot_angle) - unrot_q1 * np.sin(rot_angle)
            # rot_q1 = unrot_i1 * np.sin(rot_angle) + unrot_q1 * np.cos(rot_angle)

            rot_res = np.array([rot_i0, rot_i1])
        else:
            rot_angle = self.config["eval"]["proj_rotation"][self.config["eval"]["measure_state"]]
            rot_res = avgi_arr * np.cos(rot_angle) - avgq_arr * np.sin(rot_angle)
            print(f'rot_angle: {rot_angle}')
            print(f'rot_res: {rot_res}')
            # result_amps = np.sqrt(np.power(avgi_arr, 2) + np.power(avgq_arr, 2))
        # print(f'result_amps: {result_amps}')
        measure_probs = (rot_res - eval_dict['background_amp'])
        # print(f'measure_probs: {measure_probs}')
        hamiltonian_elems = measure_probs * eval_dict['coeffs']
        # print(f'hamiltonian_elems: {hamiltonian_elems}')
        res_sum = np.sum(hamiltonian_elems) / eval_dict['norm_amp']

        # Save data
        if save:
            data_dict = {"res_sum": res_sum}
            self.save_data(data=data_dict, name="arb_evaluate_expect", file_name=self.config["expt"]["hdf5"])

        return res_sum

    def gauss_amps(self, ch, sigma, length):
        """Adds a Gaussian to the envelope library.
                The envelope will peak at length/2.
                Duration units depend on the program type: tProc v1 programs use integer number of fabric clocks, tProc v2 programs use float us.

                Parameters
                ----------
                :param ch: Generator channel
                :param name: Name of the envelope
                :param sigma: Standard deviation of the Gaussian [clock ticks]
                :param length: Total envelope length [clock ticks]
                """

        gencfg = self.soccfg['gens'][ch]
        maxv = gencfg['maxv'] * gencfg['maxv_scale']
        samps_per_clk = gencfg['samps_per_clk']

        # convert to number of samples
        sigma *= samps_per_clk
        length *= samps_per_clk

        x_pts = np.arange(length)
        gauss_func = self.gauss(A=maxv, mu=(length / 2) - 0.5, sigma=sigma, C=0)

        y_pts = gauss_func(x_pts).astype(int)

        return y_pts

    def add_test_waveform(self, ch, *args):
        """
        Adds a waveform to the specified channel.

        :param ch: Channel to play pulse on
        :param args: Contains either a dict containing the name, I, Q, gain, freq, phase, and t, or variables defining that
        :return:
        """

        if "pulses" not in self.config[self.channel_ids[ch]].keys():
            self.config[self.channel_ids[ch]]["pulses"] = []

        if len(args) == 1:
            self.config[self.channel_ids[ch]]["pulses"].append(args[0])
        else:
            self.config[self.channel_ids[ch]]["pulses"].append({
                "name": args[0],
                "I": args[1],
                "Q": args[2],
                "gain": args[3],
                "freq": args[4],
                "phase": args[5],
                "t": args[6]
            })

    def set_test_waveform(self, ch, pulse_id, *args):
        """
        Can break easily, use with caution

        :param ch: Channel to play pulse on
        :param pulse_id: Pulse ID to modify, careful when keeping track of this  TODO: add cases to manage incorrect IDs
        :param args: Contains either a dict containing the name, I, Q, gain, freq, phase, and t, or variables defining that
        :return:
        """

        if len(args) == 1:
            self.config[self.channel_ids[ch]]["pulses"][pulse_id] = args[0]
        else:
            self.config[self.channel_ids[ch]]["pulses"][pulse_id] = {
                "name": args[0],
                "I": args[1],
                "Q": args[2],
                "gain": args[3],
                "freq": args[4],
                "phase": args[5],
                "t": args[6]
            }

    def clear_test_waveform(self, chs=None):
        """
        Clears all test waveforms from all channels, use when you want to test a new set of waveforms. Alternatively, specify the specific channel(s) that you want to clear

        :param chs:
        :return:
        """
        if chs is None:
            for channel_name in self.config.keys():
                channel = self.config[channel_name]
                if "pulses" in channel:
                    self.config[channel_name]['pulses'] = []
        elif type(chs) == int:
            self.config[self.channel_ids[chs]]["pulses"] = []
        else:
            for ch in chs:
                self.config[self.channel_ids[ch]]["pulses"] = []

    def pulse_view(self, title="", n=0, save=True):
        """
        Plots graph for the identified observable using the currently inputted waveforms.

        :param title:
        :param n: Number of the observable to plot
        :param save: Whether to save the plot or not
        :return:
        """
        self.config["observable_pulses"] = self.observable_dicts[n]

        # Create QiCK program object
        prog_obj = ArbitraryEvaluate(soc=self.soc, soccfg=self.soccfg, config=self.config)

        pulse_obj = PulseSequences(prog_obj.create(), loop_num=1)
        pulse_obj.plotPulsesGain(title=title, save=save)

    def construct_waveform(self, soccfg, base_func, length, num_segs):
        """

        :param soccfg:
        :param base_func:
        :param length:
        :param num_segs:
        """

class ConstructWaveform:
    def __init__(self, soccfg, base_func, length, num_segs):
        """

        :param soccfg:
        :param base_func:
        :param length:
        :param num_segs:
        """

        self.soccfg = soccfg
        self.base_func = base_func
        self.length = length
        self.num_segs = num_segs