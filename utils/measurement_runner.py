"""
File for executing setups as defined in measurements.py
"""
from measurements import Measurements
from qick import QickConfig
import Pyro4
from utils.pulse_configs import *
import numpy as np

# ==============================================
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
# ==============================================

# Connection check
# print(soccfg)

# ==============================================
# Experiment name reference

"""
expt_name_list = [
    "ge_g_contrast",
    "ef_g_contrast",
    "ef_e_contrast",
    "gf_e_contrast",
    "ge_projection_angle",
    "ef_projection_angle",
    "gf_projection_angle",
    "c0g_contrast",
    "c1g_contrast",
    "c2g_contrast",
    "single_shot",
    "SNAP_calibration",
    "state_characterization",
    "ge_power_rabi",
]
"""

# Control active storage cavity
config_dict["storage"] = config_dict["storage_A"]

# Experiment to run
expt_name = "single_shot"
print(f"\nRunning Experiment:\n{expt_name}")

# Open pulse viewer
debug = True
n_reps = 3  # number of experiments to view

# Run program
run = True  # Whether the program should run, only set to False if testing pulse grapher

# Data file
hdf5_file_name = "arb_eval_test"

measure_progs = Measurements(soc=soc, soccfg=soccfg, config_dict=config_dict, hdf5_file_name=hdf5_file_name)

match expt_name:
    case "ge_g_contrast":
        measure_progs.ge_g_contrast(run=run, debug=debug)
    case "ef_g_contrast":
        measure_progs.ef_g_contrast(r1un=run, debug=debug)
    case "ef_e_contrast":
        measure_progs.ef_e_contrast(run=run, debug=debug)
    case "gf_e_contrast":
        measure_progs.gf_e_contrast(run=run, debug=debug)
    case "ge_projection_angle":
        measure_progs.ge_projection_angle(run=run, debug=debug)
    case "ef_projection_angle":
        measure_progs.ef_projection_angle(run=run, debug=debug)
    case "gf_projection_angle":
        measure_progs.gf_projection_angle(run=run, debug=debug)
    case "c0g_contrast":
        measure_progs.c0g_contrast(run=run, debug=debug)
    case "c1g_contrast":
        measure_progs.c1g_contrast(run=run, debug=debug)
    case "c2g_contrast":
        measure_progs.c2g_contrast(run=run, debug=debug)
    case "single_shot":
        measure_progs.single_shot(run=run, debug=debug)
    case "SNAP_calibration":
        measure_progs.SNAP_calibration(run=run, debug=debug)
    case "state_characterization":
        measure_progs.state_characterization(run=run, debug=debug)
    case "ge_power_rabi":
        measure_progs.ge_power_rabi(run=run, debug=debug)