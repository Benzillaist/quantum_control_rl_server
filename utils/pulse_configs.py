"""
Pulse configuration file for cavity measurements
"""

qubit_dict = {
    "ch": 2,  # DAC channel index
    "phase": 0,  #
    "nqz": 2,  # nyquist zone
    "gain": 1000,  # probe gain [DAC units]
    "length": 0,  # probe length [us]
    "g_avg": 0,  # g state average
    "e_avg": 0,  # e state average
    "f_avg": 0,  # f state average
    "ge": {
        "freq": 5325.285,  # [MHz]
        "sigma": 0.02,  # [us]
        "pi_gain": 17225,  # [DAC units]
        "half_pi_gain": 17225 // 2,  # [DAC units]
        "selective_sigma": 0.25,  # [us]
        "selective_pi_gain": 1434,  # [DAC units]
        "threshold": None,  # int
        "projection_rotation": -1.3617941058026162  # [rads]
    },
    "ef": {
        "freq": 5118.613,  # [MHz]
        "pi_gain": 5623,  # [DAC units]
        "sigma": 0.05,  # [us]
        "selective_pi_gain": 743,  # [DAC units]
        "selective_sigma": 0.30,  # [us]
        "threshold": None,  # int
    },
    "freqs": {
        "g": {
            "ge": 5323.023,
            "ef": 5116.154,
            "gf": 0,
        },
        "e": {
            "ge": 5324.166,
            "ef": 5117.330,
            "gf": 0,
        },
        "f": {
            "ge": 0,
            "ef": 0,
            "gf": 0,
        }
    },
    "selective_freqs": {
        "g": {
            "ge": 5322.536,
            "ef": 5117.984,
            "gf": 0,
        },
        "e": {
            "ge": 5325.058,
            "ef": 5118.002,
        },
            "gf": 0,
        "f": {
            "ge": 0,
            "ef": 0,
            "gf": 0,
        }
    },
}

res_dict = {
    "ch": 0,  # DAC channel index
    "phase": 0,  #
    "nqz": 1,  # nyquist zone
    "freq": 52.589,  # [MHz]
    "freqs": {
        "g": 50.15,  # [MHz]
        "e": 55.789,  # [MHz]
        "f": 0,  # [MHz]
    },
    "gain": 19030,  # [DAC units]
    "length": 1.341,  # [us]
    "sigma": 0  # [us]
}

storage_dict_A = {
    "ch": 4,  # DAC channel index
    "phase": 0,  #
    "nqz": 2,  # nyquist zone
    "freq": 6093.953,  # [MHz]
    "gain": 10000,  # [DAC units]
    "sigma": 0.01,  # [us]
    "chi": -5.23600,  # [MHz]
    "alpha_amp": 7080,  # [DAC units]
}

storage_dict_B = {
    "ch": 4,  # DAC channel index
    "phase": 0,  #
    "nqz": 2,  # nyquist zone
    "freq": 6091.2,  # [MHz]
    "gain": 10000,  # [DAC units]
    "sigma": 0.01,  # [us]
    "chi": -5.23600,  # [MHz]
    "alpha_amp": 7080,  # [DAC units]
}

readout_dict = {
    "ch": 0,  # ADC channel index
    "phase": 0,  #
    "adc_trig_offset": 309,  # [clock ticks]
    "length": 0.8148,  # Window length [us]
    "relax_delay": 700  # [us]
}

eval_dict = {
    "coeffs": [1],  # Coefficients of the observables to be measured, in the order that the observable
    "norm_amp": {  # Normalization constant, total voltage change between a 0% probability state and 100% probability
        # state (essentially max contrast possible)
        "g": 1,
        "e": 1,
        "f": 1,
    },
    "background_amp": 0,
    "measure_state": "g",
    "proj_rotation": {
        "g": 1.2248337636324569,
        "e": -0.3377342412366,
        "f": 0
    },
}

config_dict = {
    "res": res_dict,
    "readout": readout_dict,
    "qubit": qubit_dict,
    "storage_A": storage_dict_A,
    "storage_B": storage_dict_B,
    "eval": eval_dict,
}


# Observable pulses
def pi_pulse(qubit_name="qubit", rabi_states="ge", meas_state='g', selective=False, ch=None, nqz=None, name=None,
             gain=None, freq=None, phase=None, sigma=None, t=None):
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
    if freq is None:
        freq = config_dict[qubit_name][sel_str + "freqs"][meas_state][rabi_states]
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
        "gain": gain,  # Gain of pulse [DAC units]
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

