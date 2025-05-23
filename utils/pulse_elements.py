import json

# import config_dict
with open("../../utils/pulse_configs.py") as f:
    data = f.read()

config_dict = json.loads(data)

# Observable pulses
def pi_pulse(qubit_name="qubit", rabi_states="ge", meas_state='g', selective=False, ch=None, nqz=None, name=None,
             gain=None, gain_mult=None, freq=None, phase=None, sigma=None, t=None):
    """
    Plays gaussian pulse, defaults to GE pi pulse measured at |g>

    :param qubit_name:
    :param rabi_states:
    :param meas_state:
    :param selective:
    :param ch:
    :param nqz:
    :param name:
    :param gain:
    :param freq:
    :param phase:
    :param sigma:
    :param t:
    :return:
    """

    sel_str = ""
    if selective:
        sel_str = "selective_"
    if ch is None:
        ch = config_dict[qubit_name]["ch"]
    if nqz is None:
        nqz = config_dict[qubit_name]["nqz"]
    if name is None:
        name = sel_str + rabi_states + "-pi-pulse"
    if gain is None:
        gain = config_dict[qubit_name][rabi_states][sel_str + "pi_gain"]
    if gain_mult is None:
        gain_mult = 1
    if freq is None:
        freq = config_dict[qubit_name]["freqs"][meas_state][rabi_states]
    if phase is None:
        phase = 0
    if sigma is None:
        sigma = config_dict[qubit_name][rabi_states][sel_str + "sigma"]
    if t is None:
        t = 'auto'

    return {
        "ch": ch,  # Channel to play pulse on
        "nqz": nqz,  # Nyquist zone of frequency
        "name": name,  # Name of pulse, should be unique across program
        "gain": gain * gain_mult,  # Gain of pulse [DAC units]
        "freq": freq,  # Frequency to play pulse at [MHz]
        "phase": phase,  # Phase of pulse [degrees]
        "sigma": sigma,  # Standard deviation of Gaussian [us]
        "t": t  # Time at which pulse is sent [us]
    }


# displacement pulse

def displacement(cavity="storage_A", alpha=1, ch=None, nqz=None, name=None, gain=None, freq=None, phase=None,
                 sigma=None, t=None):
    """
    Plays cavity displacement pulse in the form of a Gaussian, defaults to a displacement creating a coherent state size of 1

    :param cavity:
    :param alpha:
    :param ch:
    :param nqz:
    :param name:
    :param gain:
    :param freq:
    :param phase:
    :param sigma:
    :param t:
    :return:
    """

    if ch is None:
        ch = config_dict[cavity]["ch"]
    if nqz is None:
        nqz = config_dict[cavity]["nqz"]
    if name is None:
        name = "cavity_displacement"
    if gain is None:
        gain = int(config_dict[cavity]["alpha_amp"] * alpha)
    if freq is None:
        freq = config_dict[cavity]["freq"]
    if phase is None:
        phase = 0
    if sigma is None:
        sigma = config_dict[cavity]["sigma"]
    if t is None:
        t = 'auto'

    return {
        "ch": ch,  # Channel to play pulse on
        "nqz": nqz,  # Nyquist zone of frequency
        "name": name,  # Name of pulse, should be unique across program
        "gain": gain,  # Gain of pulse [DAC units]
        "freq": freq,  # Frequency to play pulse at [MHz]
        "phase": phase,  # Phase of pulse [degrees]
        "sigma": sigma,  # Standard deviation of Gaussian [us]
        "t": t  # Time at which pulse is sent [us]
    }