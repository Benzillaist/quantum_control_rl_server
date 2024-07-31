"""
Class for multiply functions
"""

import h5py
import matplotlib.pyplot as plt
import matplotlib.figure
from datetime import datetime
from os import path, makedirs
from yaml import dump
from typing import Dict
import numpy as np
from math import floor
from matplotlib.lines import Line2D


class Operations(object):
    def __init__(self, soccfg, config: Dict):
        self.soccfg = soccfg
        self.config = config
        self.argsmat = np.array([[8.64723257, -92.98015216],
                        [8.6442681, -94.35555672],
                        [8.6476505, -94.4645287],
                        [8.66617373, -96.30639282],
                        [8.6646043, -98.1560989],
                        [8.67368759, -99.27055148],
                        [8.66712157, -100.78789311],
                        [8.67361489, -101.19631196],
                        [8.66708256, -101.84029674],
                        [8.69770714, -101.87801716],
                        [8.66995913, -101.98416292],
                        [8.69669599, -102.9978097],
                        [8.68061229, -103.98073117],
                        [8.66211427, -104.9654398]])

    def save_figure(self, name: str, root_folder: str = r'C:\_Data\images', **kwargs) -> None:
        """

        :param name:
        :param root_folder:
        :param kwargs:
        :return:
        """
        # Get running time string
        date_str = datetime.now().strftime("%Y%m%d")
        time_str = datetime.now().strftime("%H%M%S")

        folder_path = path.join(root_folder, f"{date_str}")
        file_path = path.join(folder_path, f"{time_str}_{name}.png")

        makedirs(folder_path, exist_ok=True)

        plt.savefig(file_path, **kwargs)

        # print(f"\n{file_path} has been saved.\n")

    def save_data(self, data: Dict, name: str, file_name: str, root_folder: str = r'C:\_Data\datasets') -> None:
        """
        :param data:
        :param name:
        :param file_name:
        :param root_folder:

        :return:
        """
        # Get running time string
        date_str = datetime.now().strftime("%Y%m%d")
        time_str = datetime.now().strftime("%H%M%S")

        file_path = path.join(root_folder, f"{file_name}.h5")

        makedirs(root_folder, exist_ok=True)

        def group_exists(h5_file, group_name):
            return group_name in h5_file

        with h5py.File(file_path, 'a') as file:
            if not group_exists(file, date_str):
                date_group = file.create_group(date_str)
            else:
                date_group = file[date_str]

            expt_str = f"{time_str}_{name}"
            if not group_exists(date_group, expt_str):
                expt_group = date_group.create_group(expt_str)
            elif not group_exists(date_group, expt_str + "_1"):
                expt_group = date_group.create_group(expt_str + "_1")
            else:
                expt_group = date_group.create_group(expt_str + "_2")

            for key, value in data.items():
                expt_group.create_dataset(key, data=value)

        # print(f"\n{file_path}:{expt_str} has been saved.\n")

    def save_configuration(self, config: Dict, name: str, root_folder: str = r'C:\_Data\settings') -> None:
        """
        Save configuration dictionary as a yaml file

        :param config: Configuration data
        :param name: Experiment name
        :param root_folder:

        :return:
        """
        # Get running time string
        date_str = datetime.now().strftime("%Y%m%d")
        time_str = datetime.now().strftime("%H%M%S")

        # Create folder
        folder_path = path.join(root_folder, f"{date_str}")
        file_path = path.join(folder_path, f"{time_str}_{name}.yml")

        makedirs(folder_path, exist_ok=True)

        # Save configuration in a yaml file
        with open(file_path, "w") as file:
            dump(config, file)

        # print(f"\n{file_path} has been saved.\n")

    def load_configuration(self):
        pass

    def dB_to_gain(self, dB, freq):
        ind = int(floor(freq / 500)) - 1
        if ind > 13:
            ind = 13
        return int(np.e ** ((dB - self.argsmat[ind, 1]) / self.argsmat[ind, 0]))

    def gain_to_dB(self, gain, freq):
        ind = int(floor(freq / 500)) - 1
        if ind > 13:
            ind = 13
        return self.argsmat[ind, 0] * np.log(gain) + self.argsmat[ind, 1]

    def gauss(self, A, mu, sigma, C):
        def gauss_helper(x):
            return (A * np.exp(-(x - mu) ** 2 / sigma ** 2)) + C

        return gauss_helper

    def pulseSequences(self, prog_obj):
        prog = prog_obj.dump_prog()
        soccfg = prog_obj.soccfg
        readout_chs = soccfg['readouts']
        generator_chs = prog_obj.soccfg['gens']
        asm = prog['prog_list']
        regs = np.zeros((8, 32), dtype=np.int32)
        # event struct: {channel::int, freq::int, phase::int, addr::int, gain::int, mode::int, time::int
        stream = np.array([])
        loops = {}
        stack = []  # TODO: add failure conditions later (pg. 3)
        data_memory = np.zeros((32), dtype=np.int32)

        toff = 0
        i = 0
        orig_freqs = {}
        while i < len(asm):
            inst = asm[i]
            argt = inst['args']
            print(inst)
            if 'label' in inst:
                loops[inst['label']] = i
            if inst['name'] == 'pushi':
                stack.append(regs[argt[0], argt[1]])
                regs[argt[0], argt[2]] = argt[3]
            elif inst['name'] == 'popi':
                regs[argt[0], argt[1]] = stack.pop()
            elif inst['name'] == 'mathi':
                ret = 0
                if argt[3] == '+':
                    ret = regs[argt[0], argt[2]] + int(argt[4])
                    if str((32 * argt[0]) + argt[2]) in orig_freqs:
                        # print("UPDATING: " + str(orig_freqs[str((32 * argt[0]) + argt[2])]) + " to: " + str(orig_freqs[str((32 * argt[0]) + argt[2])] + int(argt[4])))
                        orig_freqs[str((32 * argt[0]) + argt[1])] = orig_freqs[
                                                                        str((
                                                                                        32 *
                                                                                        argt[
                                                                                            0]) +
                                                                            argt[
                                                                                2])] + int(
                            argt[4])
                elif argt[3] == '-':
                    ret = regs[argt[0], argt[2]] - int(argt[4])
                    if str((32 * argt[0]) + argt[2]) in orig_freqs:
                        orig_freqs[str((32 * argt[0]) + argt[1])] = orig_freqs[
                                                                        str((
                                                                                        32 *
                                                                                        argt[
                                                                                            0]) +
                                                                            argt[
                                                                                2])] - int(
                            argt[4])
                elif argt[3] == '*':
                    ret = regs[argt[0], argt[2]] * int(argt[4])
                    if str((32 * argt[0]) + argt[2]) in orig_freqs:
                        orig_freqs[str((32 * argt[0]) + argt[1])] = orig_freqs[
                                                                        str((
                                                                                        32 *
                                                                                        argt[
                                                                                            0]) +
                                                                            argt[
                                                                                2])] * int(
                            argt[4])
                regs[argt[0], argt[1]] = ret
            elif inst['name'] == 'seti':
                stream = np.append(stream, {'type': 'set',
                                            'channel': argt[0],
                                            'out': regs[argt[1], argt[2]],
                                            'time': (toff + argt[3]) / 430.08,
                                            'toff': toff / 430.08})
            elif inst['name'] == 'synci':
                toff += argt[0]
            elif inst['name'] == 'waiti':
                stream = np.append(stream, {'type': 'measure',
                                            'channel': argt[0],
                                            'end time': (argt[1] / 602.112),
                                            # + (toff/430.08) + ,
                                            'toff': toff / 430.08})
            elif inst['name'] == 'bitwi':
                ret = 0
                if argt[3] == '&':
                    ret = regs[argt[0], argt[2]] & int(argt[4])
                elif argt[3] == '|':
                    ret = regs[argt[0], argt[2]] | int(argt[4])
                elif argt[3] == '^':
                    ret = regs[argt[0], argt[2]] ^ int(argt[4])
                elif argt[3] == '~':
                    ret = ~ int(argt[4])
                elif argt[3] == '<<':
                    ret = regs[argt[0], argt[2]] << int(argt[4])
                elif argt[3] == '>>':
                    ret = regs[argt[0], argt[2]] >> int(argt[4])
                regs[argt[0], argt[1]] = ret
            elif inst['name'] == 'memri':
                regs[argt[0], argt[1]] = data_memory[argt[2]]
            elif inst['name'] == 'memwi':
                data_memory[argt[2]] = regs[argt[0], argt[1]]
            elif inst['name'] == 'regwi':
                regs[argt[0], argt[1]] = int(argt[2])
                if 'comment' in inst:
                    if inst['comment'] is not None:
                        if 'freq' in inst['comment']:
                            orig_freqs[str((32 * argt[0] + argt[1]))] = int(
                                inst['comment'][7:])
            elif inst['name'] == 'loopnz':
                if regs[argt[0], argt[1]] != 0:
                    regs[argt[0], argt[1]] -= 1
                    i = loops[argt[2]] - 1
            elif inst['name'] == 'condj':
                if argt[2] == '>':
                    if regs[argt[0], argt[1]] > regs[argt[0], argt[3]]:
                        i = loops[argt[4]] - 1
                elif argt[2] == '>=':
                    if regs[argt[0], argt[1]] >= regs[argt[0], argt[3]]:
                        i = loops[argt[4]] - 1
                elif argt[2] == '<':
                    if regs[argt[0], argt[1]] < regs[argt[0], argt[3]]:
                        i = loops[argt[4]] - 1
                elif argt[2] == '<=':
                    if regs[argt[0], argt[1]] <= regs[argt[0], argt[3]]:
                        i = loops[argt[4]] - 1
                elif argt[2] == '==':
                    if regs[argt[0], argt[1]] == regs[argt[0], argt[3]]:
                        i = loops[argt[4]] - 1
                elif argt[2] == '!=':
                    if regs[argt[0], argt[1]] != regs[argt[0], argt[3]]:
                        i = loops[argt[4]] - 1
            elif inst['name'] == 'end':
                stream = np.append(stream, {'type': 'end'})
            elif inst['name'] == 'math':
                ret = 0
                if argt[3] == '+':
                    ret = regs[argt[0], argt[2]] + regs[argt[0], argt[4]]
                    if str((32 * argt[0]) + argt[1]) in orig_freqs:
                        # print("UPDATING: " + str(orig_freqs[str((32 * argt[0]) + argt[2])]) + " to: " + str(orig_freqs[str((32 * argt[0]) + argt[2])] + int(argt[4])))
                        orig_freqs[str((32 * argt[0]) + argt[1])] = orig_freqs[
                                                                        str((
                                                                                        32 *
                                                                                        argt[
                                                                                            0]) +
                                                                            argt[
                                                                                2])] + int(
                            regs[argt[0], argt[4]])
                elif argt[3] == '-':
                    ret = regs[argt[0], argt[2]] - regs[argt[0], argt[4]]
                    if str((32 * argt[0]) + argt[1]) in orig_freqs:
                        # print("UPDATING: " + str(orig_freqs[str((32 * argt[0]) + argt[2])]) + " to: " + str(orig_freqs[str((32 * argt[0]) + argt[2])] + int(argt[4])))
                        orig_freqs[str((32 * argt[0]) + argt[1])] = orig_freqs[
                                                                        str((
                                                                                        32 *
                                                                                        argt[
                                                                                            0]) +
                                                                            argt[
                                                                                2])] - int(
                            regs[argt[0], argt[4]])
                elif argt[3] == '*':
                    ret = regs[argt[0], argt[2]] * regs[argt[0], argt[4]]
                    if str((32 * argt[0]) + argt[1]) in orig_freqs:
                        # print("UPDATING: " + str(orig_freqs[str((32 * argt[0]) + argt[2])]) + " to: " + str(orig_freqs[str((32 * argt[0]) + argt[2])] + int(argt[4])))
                        orig_freqs[str((32 * argt[0]) + argt[1])] = orig_freqs[
                                                                        str((
                                                                                        32 *
                                                                                        argt[
                                                                                            0]) +
                                                                            argt[
                                                                                2])] * int(
                            regs[argt[0], argt[4]])
                regs[argt[0], argt[1]] = ret
            elif inst['name'] == 'set':
                page = argt[1]
                # print(orig_freqs)
                # print(str((32 * page) + argt[2]))
                # print(orig_freqs[str((32 * page) + argt[2])])
                stream = np.append(stream, {'type': 'pulse',
                                            'channel': argt[0] - 1,
                                            'freq': orig_freqs[
                                                        str((32 * page) + argt[
                                                            2])] / 624152.38,
                                            # 'freq': ((int(inst['comment'][7:]) / 624152.38) if 'freq' in inst['comment'] else (500000000 * int(bin(regs[page, argt[2]])[-16:], 2) / 65535)) if 'comment' in inst else (500000000 * int(bin(regs[page, argt[2]])[-16:], 2) / 65535),
                                            # 'freq': 500000000 * int(bin(regs[page, argt[2]])[-16:], 2) / 65535,
                                            'phase': 2 * np.pi * int(
                                                bin(regs[page, argt[3]])[-16:],
                                                2) / 65535,
                                            'addr': None if argt[
                                                                4] == 0 else int(
                                                bin(regs[page, argt[4]])[-16:],
                                                2),
                                            'addr_length_clk': int(
                                                bin(regs[page, argt[6]])[-12:],
                                                2) * generator_chs[argt[
                                                                       0] - 1][
                                                                   'samps_per_clk'],
                                            'gain': int(
                                                bin(regs[page, argt[5]])[-16:],
                                                2),
                                            # 'gain': BitArray(bin=bin(regs[page, argt[5]])[-16:]).int,
                                            'length': int(
                                                bin(regs[page, argt[6]])[-12:],
                                                2) / 430.08,
                                            'time': (toff + regs[
                                                page, argt[7]]) / 430.08,
                                            'toff': toff / 430.08})
            elif inst['name'] == 'sync':
                toff += regs[argt[0], argt[1]]
            elif inst['name'] == 'read':
                stream = np.append(stream, {'type': 'read',
                                            'page': argt[0],
                                            'reg': argt[1],
                                            'time': (toff + regs[
                                                page, argt[7]]) / 430.08,
                                            'toff': toff / 430.08
                                            })
            elif inst['name'] == 'waiti':
                stream = np.append(stream, {'type': 'measure',
                                            'channel': argt[0],
                                            'end time': (regs[argt[0], argt[
                                                1]] / 602.112),
                                            # + (toff/430.08) + ,
                                            'toff': toff / 430.08})
            elif inst['name'] == 'bitw':
                ret = 0
                if argt[3] == '&':
                    ret = regs[argt[0], argt[2]] & int(regs[argt[0], argt[4]])
                elif argt[3] == '|':
                    ret = regs[argt[0], argt[2]] | int(regs[argt[0], argt[4]])
                elif argt[3] == '^':
                    ret = regs[argt[0], argt[2]] ^ int(regs[argt[0], argt[4]])
                elif argt[3] == '~':
                    ret = ~ int(regs[argt[0], argt[4]])
                elif argt[3] == '<<':
                    ret = regs[argt[0], argt[2]] << int(regs[argt[0], argt[4]])
                elif argt[3] == '>>':
                    ret = regs[argt[0], argt[2]] >> int(regs[argt[0], argt[4]])
                regs[argt[0], argt[1]] = ret
            elif inst['name'] == 'memri':
                regs[argt[0], argt[1]] = data_memory[regs[argt[0], argt[2]]]
            elif inst['name'] == 'memwi':
                data_memory[regs[argt[0], argt[2]]] = regs[argt[0], argt[1]]
            # elif inst['name'] == 'memwi':
            #     stream = np.append(stream, {'type': 'write',
            #                                 'page': argt[0],
            #                                 'register': argt[1],
            #                                 'address': argt[2],
            #                                 'toff': toff/430.08})
            else:
                print("Unrecognized command")
            i += 1

        # regwi - done
        # bitwi - done
        # mathi - done
        # synci - done
        # set - in progress
        # seti - done
        # waiti - done
        # memwi - done
        # loopnz - done
        # end - done
        return stream

    def gaus(self, x, u, s):
        return np.exp(-np.power((x - u), 2) / (2 * np.power(s, 2)))

    def colors(self, ch):
        c = "black"
        if ch == 1:
            c = 'blue'
        elif ch == 2:
            c = 'green'
        elif ch == 3:
            c = 'red'
        elif ch == 4:
            c = 'orange'
        elif ch == 5:
            c = 'yellow'
        elif ch == 6:
            c = 'pink'
        elif ch == 7:
            c = 'brown'
        return c

    def plotPulsesGain(self, prog_obj, pulses, title=""):
        prog = prog_obj.dump_prog()
        soccfg = prog_obj.soccfg
        readout_chs = soccfg['readouts']
        generator_chs = prog_obj.soccfg['gens']
        asm = prog['prog_list']
        envelopes = prog['envelopes']

        # sets up the waveforms for the table to hold
        table = []
        for i in range(7):
            table.append(np.array([]))
        for (i, ch) in enumerate(envelopes):
            for envelope in ch:
                table[i] = np.append(table[i], ch[envelope]['data'][:, 0])

        # creates list of all channels that are used:
        used_generator_chs = []
        used_readout_chs = []
        for i, event in enumerate(pulses):
            if event['type'] == 'pulse':
                if event['channel'] not in used_generator_chs:
                    used_generator_chs.append(event['channel'])
            elif event['type'] == 'measure':
                if event['channel'] not in used_readout_chs:
                    used_readout_chs.append(event['channel'])
        used_generator_chs.sort()
        used_readout_chs.sort()
        num_generator_chs = len(used_generator_chs)
        num_readout_chs = len(used_readout_chs)
        num_used_chs = num_generator_chs + num_readout_chs

        # set up plotting axes
        axs = np.empty((num_used_chs), dtype=plt.Axes)
        fig = plt.figure()
        ax1 = plt.subplot(num_used_chs, 1, 1)
        ax1.margins(0.0, 0.2)
        if len(used_generator_chs) > 0:
            plt.title("DAC Channel " + str(used_generator_chs[0]))
        elif len(used_readout_chs) > 0:
            plt.title("ADC Channel " + str(used_generator_chs[0]))
        else:
            return 0

        for i in range(1, len(used_generator_chs)):
            axt = plt.subplot(num_used_chs, 1, i + 1, sharex=ax1)
            plt.title("DAC Channel " + str(used_generator_chs[i]))
            axt.margins(0.0, 0.2)
            axs[i] = axt
        for i in range(len(used_readout_chs)):
            axt = plt.subplot(num_used_chs, 1, num_generator_chs + i + 1,
                              sharex=ax1)
            plt.title("ADC Channel " + str(used_readout_chs[i]))
            axt.margins(0.0, 0.2)
            axs[i + num_generator_chs] = axt
        t = 0

        # setup frequency arrays
        # list for holding all frequencies for color reasons
        freqs = []
        # list of lists for holding all frequencies for each channel -- used for plotting reasons
        ch_freqs = []
        for i in range(num_used_chs):
            ch_freqs.append([])

        for i, event in enumerate(pulses):
            if event['type'] == 'pulse':
                ch = event['channel']
                ch_index = used_generator_chs.index(ch)
                freq = event['freq']

                t = event['time']
                if 'length' in event:
                    # set color based on frequency
                    if round(freq, 2) not in freqs:
                        freqs.append(round(freq, 2))
                    if round(freq, 2) not in ch_freqs[ch_index]:
                        ch_freqs[ch_index].append(round(freq, 2))
                    c = self.colors(freqs.index(round(freq, 2)))

                    # setup graph
                    axs[ch_index] = plt.subplot(num_used_chs, 1, ch_index + 1,
                                                sharex=ax1)
                    plt.title("DAC Channel " + str(ch))
                    plt.margins(0.0, 0.2)

                    # doing it correctly
                    if event['addr'] is not None:
                        # ADD WAVEFORM IN TABLE
                        table_read_start = int(event['addr']) << 4
                        pulseTimes = np.linspace(t, t + event['length'],
                                                 event['addr_length_clk'])
                        pulseAmps = table[ch][
                                    table_read_start:table_read_start + event[
                                        'addr_length_clk']] * event['gain'] / \
                                    generator_chs[ch]['maxv']
                        plt.plot(pulseTimes, pulseAmps, color=c,
                                 label=f'Pulse @ {round(self.gain_to_dB(event["gain"], freq), 2)} DAC @ {round(freq, 2)} dBm')
                    else:
                        # FLAT TOP
                        pulseTimes = np.linspace(t, t + event['length'], 100)
                        gainRange = np.linspace(0, event['gain'], 10)
                        plt.plot(np.ones(len(gainRange)) * t, gainRange,
                                 color=c)
                        plt.plot(pulseTimes,
                                 np.ones(len(pulseTimes)) * event['gain'],
                                 color=c,
                                 label=f'Pulse @ {round(self.gain_to_dB(event["gain"], freq), 2)} DAC @ {round(freq, 2)} dBm')
                        # t = t + event['length']
                        plt.plot(
                            np.ones(len(gainRange)) * (t + event['length']),
                            gainRange, color=c)
            elif event['type'] == 'measure':
                ch = event['channel']
                ch_index = used_readout_chs.index(ch) + num_generator_chs

                freq = round(prog['ro_chs'][ch]['freq'], 2)

                if round(freq, 2) not in freqs:
                    freqs.append(round(freq, 2))
                if round(freq, 2) not in ch_freqs[ch_index]:
                    ch_freqs[ch_index].append(round(freq, 2))
                c = self.colors(freqs.index(round(freq, 2)))

                axs[ch_index] = plt.subplot(num_used_chs, 1, ch_index + 1,
                                            sharex=ax1)
                plt.title("ADC Channel " + str(ch))
                trig_offset = ((event['end time'] * 430.08 * 1.4) - (
                            pulses[i - 1]['length'] * 430.08 * 1.4)) / 430.08
                measureTimes = np.linspace(event['toff'] + trig_offset,
                                           event['toff'] + trig_offset + (((
                                                                                       event[
                                                                                           'end time'] * 430.08 * 1.4) - (
                                                                                       trig_offset * 430.08)) / (
                                                                                      430.08 * 1.4)),
                                           10)
                # ax = plt.subplot(num_used_chs, 1, ch + num_generator_chs + 1, sharex=ax1)
                plt.xlim(0, event['toff'] + trig_offset + (((event[
                                                                 'end time'] * 430.08 * 1.4) - (
                                                                        trig_offset * 430.08)) / (
                                                                       430.08 * 1.4)) + 0.2)
                plt.plot(measureTimes, np.zeros(len(measureTimes)), color=c,
                         linestyle='dotted', label="Measure")
                # t = event['end time']

        # freq_labels = [(str(round(freq, 2)) + " MHz") for freq in freqs]
        # custom_lines = [Line2D([0], [0], color = colors(freqs.index(freq)), lw=4) for freq in freqs]

        for i, ax in enumerate(axs):
            freq_labels = [(str(round(freq, 2)) + " MHz") for freq in
                           ch_freqs[i]]
            custom_lines = [
                Line2D([0], [0], color=self.colors(freqs.index(freq)), lw=4) for
                freq in ch_freqs[i]]
            ax.legend(custom_lines, freq_labels)

        fig.supxlabel("Time (us)")
        fig.supylabel("Gain (DAC units)")
        plt.subplots_adjust(hspace=1.0)
        plt.suptitle(title)
        # fig.show()
        # plt.figlegend(custom_lines, freq_labels, loc=(0.9, 0.9))
        plt.show()

        # save_figure(name="pulse-seq")
