import numpy as np
import matplotlib.pyplot as plt
from time import localtime, strftime
from os import path, makedirs
from math import floor
from matplotlib.lines import Line2D


class PulseSequences:
    def __init__(self, prog_obj, loop_num = 1):
        self.prog_obj = prog_obj
        self.loop_num = loop_num
        self.argsmat = np.array([[8.64723257, -92.98015216],
         [8.64426810, -94.35555672],
         [8.64765050, -94.46452870],
         [8.66617373, -96.30639282],
         [8.66460430, -98.15609890],
         [8.67368759, -99.27055148],
         [8.66712157, -100.78789311],
         [8.67361489, -101.19631196],
         [8.66708256, -101.84029674],
         [8.69770714, -101.87801716],
         [8.66995913, -101.98416292],
         [8.69669599, -102.99780970],
         [8.68061229, -103.98073117],
         [8.66211427, -104.96543980]])
        self.pulses = self._pulseSequences()

    def dB_to_gain(self, dB, freq):
        ind = int(floor(freq / 500)) - 1
        if ind > 13:
            ind = 13
        if ind < 0:
            ind = 0
        return int(np.e ** ((dB - self.argsmat[ind, 1]) / self.argsmat[ind, 0]))

    def gain_to_dB(self, gain, freq):
        ind = int(floor(freq / 500)) - 1
        if ind > 13:
            ind = 13
        if ind < 0:
            ind = 0
        return (self.argsmat[ind, 0] * np.log(gain)) + self.argsmat[ind, 1]

    def pulseSequences(self):
        return self.pulses

    def _pulseSequences(self):
        prog = self.prog_obj.dump_prog()
        soccfg = self.prog_obj.soccfg
        readout_chs = soccfg['readouts']
        generator_chs = self.prog_obj.soccfg['gens']
        asm = prog['prog_list']

        regs = np.zeros((8, 32), dtype = np.int32)
        # event struct: {channel::int, freq::int, phase::int, addr::int, gain::int, mode::int, time::int
        stream = np.array([])
        loops = {}
        stack = [] # TODO: add failure conditions later (pg. 3)
        data_memory = np.zeros((32), dtype = np.int32)

        toff = 0
        init_toff = None
        i = 0
        loop_count = 0
        outer_loop_label = None
        orig_freqs = {}
        # keeps track of adc_trig_offset
        adc_trig_offset = 0
        while i < len(asm):
            inst = asm[i]
            argt = inst['args']
            if 'label' in inst:
                if len(loops.keys()) == 0:
                    init_toff = toff
                    outer_loop_label = inst['label']
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
                        orig_freqs[str((32 * argt[0]) + argt[1])] = orig_freqs[str((32 * argt[0]) + argt[2])] + int(argt[4])
                elif argt[3] == '-':
                    ret = regs[argt[0], argt[2]] - int(argt[4])
                    if str((32 * argt[0]) + argt[2]) in orig_freqs:
                        orig_freqs[str((32 * argt[0]) + argt[1])] = orig_freqs[str((32 * argt[0]) + argt[2])] - int(argt[4])
                elif argt[3] == '*':
                    ret = regs[argt[0], argt[2]] * int(argt[4])
                    if str((32 * argt[0]) + argt[2]) in orig_freqs:
                        orig_freqs[str((32 * argt[0]) + argt[1])] = orig_freqs[str((32 * argt[0]) + argt[2])] * int(argt[4])
                regs[argt[0], argt[1]] = ret
            elif inst['name'] == 'seti':
                if regs[argt[1], argt[2]] > 0:
                    adc_trig_offset = argt[3]
                stream = np.append(stream, {'type': 'set',
                                            'channel': argt[0],
                                            'out': regs[argt[1], argt[2]],
                                            'time': (toff + argt[3]) / 430.08,
                                            'toff': toff/430.08})
            elif inst['name'] == 'synci':
                toff += argt[0]
            elif inst['name'] == 'waiti':
                stream = np.append(stream, {'type': 'measure',
                                            'channel': argt[0],
                                            'end time': ((argt[1] - regs[argt[0], 0]) / 602.112), # + (toff/430.08) + ,
                                            'adc_trig_offset': adc_trig_offset / 602.112,
                                            'toff': toff/430.08})
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
                            orig_freqs[str((32 * argt[0] + argt[1]))] = int(inst['comment'][7:])
            elif inst['name'] == 'loopnz':
                if loop_count >= self.loop_num - 1:
                    i += 1
                    continue
                if argt[2] == outer_loop_label:
                    loop_count += 1
                    toff = init_toff
                    if regs[argt[0], argt[1]] != 0:
                        regs[argt[0], argt[1]] -= 1
                        i = loops[argt[2]] - 1
            elif inst['name'] == 'condj':
                if loop_count >= self.loop_num - 1:
                    i += 1
                    continue
                if argt[4] == outer_loop_label:
                    loop_count += 1
                    toff = init_toff
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
                        orig_freqs[str((32 * argt[0]) + argt[1])] = orig_freqs[str((32 * argt[0]) + argt[2])] + int(regs[argt[0], argt[4]])
                elif argt[3] == '-':
                    ret = regs[argt[0], argt[2]] - regs[argt[0], argt[4]]
                    if str((32 * argt[0]) + argt[1]) in orig_freqs:
                        orig_freqs[str((32 * argt[0]) + argt[1])] = orig_freqs[str((32 * argt[0]) + argt[2])] - int(regs[argt[0], argt[4]])
                elif argt[3] == '*':
                    ret = regs[argt[0], argt[2]] * regs[argt[0], argt[4]]
                    if str((32 * argt[0]) + argt[1]) in orig_freqs:
                        orig_freqs[str((32 * argt[0]) + argt[1])] = orig_freqs[str((32 * argt[0]) + argt[2])] * int(regs[argt[0], argt[4]])
                regs[argt[0], argt[1]] = ret
            elif inst['name'] == 'set':
                page = argt[1]

                stream = np.append(stream, {'type': 'pulse',
                                            'channel': argt[0] - 1,
                                            'freq': orig_freqs[str((32 * page) + argt[2])] / 624152.38,
                                            # 'freq': ((int(inst['comment'][7:]) / 624152.38) if 'freq' in inst['comment'] else (500000000 * int(bin(regs[page, argt[2]])[-16:], 2) / 65535)) if 'comment' in inst else (500000000 * int(bin(regs[page, argt[2]])[-16:], 2) / 65535),
                                            # 'freq': 500000000 * int(bin(regs[page, argt[2]])[-16:], 2) / 65535,
                                            'phase': 2 * np.pi * int(bin(regs[page, argt[3]])[-16:], 2) / 65535,
                                            'addr': None if argt[4] == 0 else int(bin(regs[page, argt[4]])[-16:], 2),
                                            'addr_length_clk': int(bin(regs[page, argt[6]])[-12:], 2) * generator_chs[argt[0] - 1]['samps_per_clk'],
                                            'gain': int(bin(regs[page, argt[5]])[bin(regs[page, argt[5]]).index('b') + 1:], 2),
                                            # 'gain': BitArray(bin=bin(regs[page, argt[5]])[-16:]).int,
                                            'length': int(bin(regs[page, argt[6]])[-12:], 2) / 430.08,
                                            'time': (toff + regs[page, argt[7]]) / 430.08,
                                            'toff': toff/430.08})
            elif inst['name'] == 'sync':
                toff += regs[argt[0], argt[1]]
            elif inst['name'] == 'read':
                stream = np.append(stream, {'type': 'read',
                                            'page': argt[0],
                                            'reg': argt[1],
                                            'time': (toff + regs[page, argt[7]]) / 430.08,
                                            'toff': toff / 430.08
                                            })
            elif inst['name'] == 'waiti':
                stream = np.append(stream, {'type': 'measure',
                                            'channel': argt[0],
                                            'end time': (regs[argt[0], argt[1]] / 602.112), # + (toff/430.08) + ,
                                            'adc_trig_offset': adc_trig_offset / 602.112,
                                            'toff': toff/430.08})
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

    def colors(self, ch):
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

    def plotPulsesGain(self, title = "", save=True):
        pulses = self.pulses
        prog = self.prog_obj.dump_prog()
        soccfg = self.prog_obj.soccfg
        readout_chs = soccfg['readouts']
        generator_chs = self.prog_obj.soccfg['gens']
        asm = prog['prog_list']
        envelopes = prog['envelopes']

        # sets up the waveforms for the table to hold
        table = []
        for i in range(7):
            table.append(np.array([]))
        for (i, envelope) in enumerate(envelopes):
            if envelope['envs'] is not None:
                for envelope_name in envelope['envs']:
                    table[i] = np.append(table[i], envelope['envs'][envelope_name]['data'][:, 0])

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
        axs = np.empty((num_used_chs), dtype = plt.Axes)
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
            axt = plt.subplot(num_used_chs, 1, i + 1, sharex = ax1)
            plt.title("DAC Channel " + str(used_generator_chs[i]))
            axt.margins(0.0, 0.2)
            axs[i] = axt
        for i in range(len(used_readout_chs)):
            axt = plt.subplot(num_used_chs, 1, num_generator_chs + i + 1, sharex = ax1)
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
                    axs[ch_index] = plt.subplot(num_used_chs, 1, ch_index + 1, sharex=ax1)
                    plt.title("DAC Channel " + str(ch))
                    plt.margins(0.0, 0.2)

                    # doing it correctly
                    if event['addr'] is not None:
                        # ADD WAVEFORM IN TABLE
                        table_read_start = int(event['addr'])<<4
                        pulseTimes = np.linspace(t, t + event['length'], event['addr_length_clk'])
                        pulseAmps = table[ch][table_read_start:table_read_start + event['addr_length_clk']] * event['gain'] / generator_chs[ch]['maxv']
                        plt.plot(pulseTimes, pulseAmps, color = c, label = f'Pulse @ {round(self.gain_to_dB(event["gain"], freq), 2)} DAC @ {round(freq, 2)} dBm')
                    else:
                        # FLAT TOP
                        pulseTimes = np.linspace(t, t + event['length'], 100)
                        gainRange = np.linspace(0, event['gain'], 10)
                        plt.plot(np.ones(len(gainRange)) * t, gainRange, color = c)
                        plt.plot(pulseTimes, np.ones(len(pulseTimes)) * event['gain'], color = c, label = f'Pulse @ {round(self.gain_to_dB(event["gain"], freq), 2)} DAC @ {round(freq, 2)} dBm')
                        # t = t + event['length']
                        plt.plot(np.ones(len(gainRange)) * (t + event['length']), gainRange, color=c)
            elif event['type'] == 'measure':
                ch = event['channel']
                ch_index = used_readout_chs.index(ch) + num_generator_chs

                freq = round(prog['ro_chs'][ch]['freq'], 2)

                if round(freq, 2) not in freqs:
                    freqs.append(round(freq, 2))
                if round(freq,2) not in ch_freqs[ch_index]:
                    ch_freqs[ch_index].append(round(freq, 2))
                c = self.colors(freqs.index(round(freq, 2)))

                axs[ch_index] = plt.subplot(num_used_chs, 1, ch_index + 1, sharex=ax1)
                plt.title("ADC Channel " + str(ch))
                trig_offset = event["adc_trig_offset"]
                measureTimes = np.linspace(event['toff'] + trig_offset, event['toff'] + trig_offset + (((event['end time'] * 430.08 * 1.4) - (trig_offset * 430.08 * 1.4)) / (430.08 * 1.4)), 10)
                plt.plot(measureTimes, np.zeros(len(measureTimes)), color = c, linestyle = 'dotted', label = "Measure")
                # t = event['end time']

        # freq_labels = [(str(round(freq, 2)) + " MHz") for freq in freqs]
        # custom_lines = [Line2D([0], [0], color = colors(freqs.index(freq)), lw=4) for freq in freqs]

        for i, ax in enumerate(axs):
            freq_labels = [(str(round(freq, 2)) + " MHz") for freq in ch_freqs[i]]
            custom_lines = [Line2D([0], [0], color = self.colors(freqs.index(freq)), lw=4) for freq in ch_freqs[i]]
            ax.legend(custom_lines, freq_labels)
            # increase axis bounds slightly
            x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
            ax.set_xlim(0 - (0.1 * x_range), ax.get_xlim()[1] + (0.1 * x_range))

        fig.supxlabel("Time (us)")
        fig.supylabel("Gain (DAC units)")
        plt.subplots_adjust(hspace = 1.0)
        plt.suptitle(title)
        # fig.show()
        # plt.figlegend(custom_lines, freq_labels, loc=(0.9, 0.9))

        if save:
            self.save_figure(name="pulse-seq")

        plt.show()

    def save_figure(self, name: str, root_folder: str = r'C:/_Data/', **kwargs) -> None:
        """

        :param name:
        :param root_folder:
        :param kwargs:
        :return:
        """
        run_time = localtime()
        file_name = path.join(root_folder, f"images/{strftime('%Y%m%d/%H%M%S' ,run_time) + '_' + name}")

        folder_name = path.split(file_name)[0]

        if not path.isdir(folder_name):
            makedirs(folder_name, exist_ok=True)

        plt.savefig(file_name, **kwargs)

        print(fr"{file_name} has been saved.")