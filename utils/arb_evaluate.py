"""
Classes for evaluating an arbitrary sequence of bosonic operators
"""

from numpy import array, append
from qick import AveragerProgram
from qick import QickConfig
from typing import Dict, Any
from utils import Operations, ConfigObj
import numpy as np


# noinspection PyTypeChecker
class ArbPulseProg(AveragerProgram):
    """
    QiCK Class: Plays an arbitrary specific pulse and measures an arbitrary bosonic observable. There are no safeguards against having really strong/long/weird pulses being played, so be careful with what you put in
    """

    def __init__(self, soccfg, config: dict):
        """
        Initialization function for QiCK class

        :param soccfg: SOC configuration
        :param config: Experiment configuration
        """
        self.soccfg = soccfg
        self.config = ConfigObj(config=config)
        self.prep_pulses = array([], dtype=dict)
        self.ctrl_pulses = array([], dtype=dict)
        self.obs_pulses = array([], dtype=dict)
        super().__init__(soccfg=soccfg, cfg=config["expt"])

    def initialize(self):
        config = self.config

        # Tracks waveform names registered to avoid multiple redefinitions of the same pulse
        waveform_names = []

        # Readout channel
        # Declare readout DAC channel
        self.declare_gen(ch=config.res.ch, nqz=config.res.nqz)

        # Convert readout frequency unit from MHz to register values (clock ticks)
        ro_freq = self.soccfg.freq2reg(f=config.res.freq, gen_ch=config.res.ch, ro_ch=config.readout.ch)

        # Set registers for readout pulse
        self.set_pulse_registers(ch=config.res.ch, freq=ro_freq, phase=0, gain=config.res.gain,
                                 length=self.soccfg.us2cycles(us=config.res.length, gen_ch=config.res.ch),
                                 style="const")

        # Configure the readout length and down-conversion frequency
        self.declare_readout(ch=config.readout.ch, length=self.soccfg.us2cycles(config.readout.length),
                             freq=config.res.freq, gen_ch=config.res.ch)

        # Loops through sub dictionaries and checks for any pulses
        channel_names = np.array(list(config.__dict__.keys()))

        # TODO: WIP
        # State preparation pulses
        for pulse in config.preparation_pulses:  # yeah, it's weird, but this is a lot easier than treating the dict as an object
            if pulse is not None:
                # Turns pulse into an object (idk if this helps, possibly remove)
                pulse = ConfigObj(pulse)
                # Set waveforms for respective channel
                waveform_name = pulse.name

                if waveform_name not in waveform_names:
                    self.add_gauss(ch=pulse.ch, name=waveform_name,
                                   sigma=self.soccfg.us2cycles(us=pulse.sigma, gen_ch=pulse.ch),
                                   length=(self.soccfg.us2cycles(us=pulse.sigma, gen_ch=pulse.ch) * 4))
                    waveform_names.append(waveform_name)

                t_temp = 0
                if type(t_temp) == "int":
                    t_temp = self.soccfg.us2cycles(pulse.t, gen_ch=pulse.ch)
                else:
                    t_temp = pulse.t
                # Adds to the list of pulses that should be played later
                self.prep_pulses = append(self.prep_pulses, ConfigObj({'ch': pulse.ch, 'waveform': waveform_name,
                                                                     'gain': pulse.gain,
                                                                     'freq': self.soccfg.freq2reg(f=pulse.freq,
                                                                                                  gen_ch=pulse.ch),
                                                                     'phase': self.soccfg.deg2reg(deg=pulse.phase,
                                                                                                  gen_ch=pulse.ch),
                                                                     't': t_temp}))


        # Control / state modification pulses
        for channel_name in channel_names[channel_names != 'dict']:
            channel = getattr(config, channel_name)
            if hasattr(channel, 'pulses'):
                channel_pulses = channel.pulses
                if len(channel_pulses) > 0:
                    # Declare DAC channel
                    self.declare_gen(ch=channel.ch, nqz=channel.nqz)

                    # Loops through pulses and sets waveforms and saves pulses
                    for pulse in channel_pulses:
                        # Set waveforms for respective channel
                        waveform_name = pulse['name']

                        if waveform_name not in waveform_names:
                            self.add_envelope(ch=channel.ch, name=waveform_name, idata=pulse['I'],
                                              qdata=pulse['Q'])
                            waveform_names.append(waveform_name)

                        # Adds to the list of pulses that should be played later
                        self.ctrl_pulses = append(self.ctrl_pulses, {'ch': channel.ch, 'waveform': waveform_name,
                                                                     'gain': pulse['gain'],
                                                                     'freq': self.soccfg.freq2reg(f=pulse['freq'],
                                                                                                  gen_ch=channel.ch),
                                                                     'phase': self.soccfg.deg2reg(deg=pulse['phase'],
                                                                                                  gen_ch=channel.ch),
                                                                     't': pulse['t']})

        for pulse in config.observable_pulses:  # yeah, it's weird, but this is a lot easier than treating the dict as an object
            if pulse is not None:
                # Turns pulse into an object (idk if this helps, possibly remove)
                pulse = ConfigObj(pulse)
                # Set waveforms for respective channel
                waveform_name = pulse.name

                if waveform_name not in waveform_names:
                    self.add_gauss(ch=pulse.ch, name=waveform_name,
                                   sigma=self.soccfg.us2cycles(us=pulse.sigma, gen_ch=pulse.ch),
                                   length=(self.soccfg.us2cycles(us=pulse.sigma, gen_ch=pulse.ch) * 4))
                    waveform_names.append(waveform_name)

                t_temp = 0
                if type(t_temp) == "int":
                    t_temp = self.soccfg.us2cycles(pulse.t, gen_ch=pulse.ch)
                else:
                    t_temp = pulse.t
                # Adds to the list of pulses that should be played later
                self.obs_pulses = append(self.obs_pulses, ConfigObj({'ch': pulse.ch, 'waveform': waveform_name,
                                                                     'gain': pulse.gain,
                                                                     'freq': self.soccfg.freq2reg(f=pulse.freq,
                                                                                                  gen_ch=pulse.ch),
                                                                     'phase': self.soccfg.deg2reg(deg=pulse.phase,
                                                                                                  gen_ch=pulse.ch),
                                                                     't': t_temp}))

        # Time for processor to generate pulses (and deal with my bad code) (430 ticks, ~1 us)
        self.sync_all(t=430)

    def body(self):
        prep_pulses = self.prep_pulses
        ctrl_pulses = self.ctrl_pulses
        obs_pulses = self.obs_pulses

        # Preparing to readout arbitrary observable
        for pulse in prep_pulses:
            # Setting pulse registers for observable preparation pulses
            self.set_pulse_registers(ch=pulse.ch, style="arb",
                                     freq=pulse.freq, phase=pulse.phase, gain=pulse.gain,
                                     waveform=pulse.waveform)
            # Play arbitrary state preparation pulse
            self.pulse(ch=pulse.ch, t=pulse.t)
            self.sync_all(self.us2cycles(0.05))  # align channels and wait 50ns

        # Plays all pulses included in the dict
        for pulse in ctrl_pulses:
            # Setting pulse registers for arbitrary control pulses
            self.set_pulse_registers(ch=pulse['ch'], style="arb",
                                     freq=pulse['freq'], phase=pulse['phase'], gain=pulse['gain'],
                                     waveform=pulse['waveform'])
            # Play arbitrary control pulse
            self.pulse(ch=pulse['ch'], t=pulse['t'])

        self.sync_all(self.us2cycles(0.05))  # align channels and wait 50ns

        # Preparing to readout arbitrary observable
        for pulse in obs_pulses:
            # Setting pulse registers for observable preparation pulses
            self.set_pulse_registers(ch=pulse.ch, style="arb",
                                     freq=pulse.freq, phase=pulse.phase, gain=pulse.gain,
                                     waveform=pulse.waveform)
            # Play arbitrary observable preparation pulse
            self.pulse(ch=pulse.ch, t=pulse.t)
            self.sync_all(self.us2cycles(0.05))  # align channels and wait 50ns

        # Readout
        self.measure(pulse_ch=self.config.res.ch,
                     adcs=[self.config.readout.ch],
                     adc_trig_offset=self.config.readout.adc_trig_offset,
                     wait=True, syncdelay=self.soccfg.us2cycles(self.config.readout.relax_delay))

    def acquire_shots(self, soc, load_pulses=True, progress=False, **kwargs):
        super().acquire(soc, load_pulses=load_pulses, progress=progress, **kwargs)
        return self.collect_shots()

    def collect_shots(self):
        shots_i = self.di_buf[0].reshape((self.config.expt.expts, self.config.expt.reps)) / self.soccfg.us2cycles(self.config.readout.length)
        shots_q = self.dq_buf[0].reshape((self.config.expt.expts, self.config.expt.reps)) / self.soccfg.us2cycles(self.config.readout.length)
        return shots_i, shots_q


class ArbitraryEvaluate(Operations):
    """
    Measure resonator frequency
    """

    def __init__(self, soc, soccfg, config: Dict):
        """

        :param soc: Soc object (from pyro4)
        :param soccfg: Soc configuration object (from pyro4)
        :param config: Configuration dictionary
        """
        super().__init__(soccfg=soccfg, config=config)
        self.soc = soc
        self.soccfg = soccfg
        self.config = config
        self.qick_obj = None

    def create(self, config=None) -> ArbPulseProg:
        """
        Create QICK object

        :param config: Updated configuration dictionary

        :return: QICK object
        """
        if config is not None:
            # Update experiment configuration
            self.config = config
            obj_temp = ArbPulseProg(soccfg=self.soccfg, config=config)
        else:
            obj_temp = ArbPulseProg(soccfg=self.soccfg, config=self.config)

        self.qick_obj = obj_temp

        return obj_temp

    def run(self, load_pulses: bool = True, progress: bool = False, all_shots: bool = False, **kwargs) -> [float, float]:
        """
        Run experiment and auto save configuration, figure and raw data

        :param load_pulses: Fire pulse sequence if true, default: true
        :param progress: Show progress bar if true, default: true
        :param kwargs:

        :return: Averaged I and Q values
        """
        # Save configuration
        # self.save_configuration(config=self.config, name="arb_execute")

        # Create and run QiCK program
        obj_temp = self.create(config=self.config)

        if all_shots:
            avgi, avgq = obj_temp.acquire_shots(soc=self.soc, load_pulses=load_pulses, progress=progress, **kwargs)

            return avgi, avgq
        else:
            avgi, avgq = obj_temp.acquire(soc=self.soc, load_pulses=load_pulses, progress=progress, **kwargs)

            return avgi[0][0], avgq[0][0]
