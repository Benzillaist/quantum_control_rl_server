from utils.operations import *
from utils.pulse_sequences import *
# from utils.opt_utils import *
from utils.pulse_configs import *
from typing import Dict

class ConfigObj(object):
    """
    Transfer configuration dictionary to object
    """
    def __init__(self, config: Dict):
        """

        :param config:
        """


        setattr(self, 'dict', config)

        for key, value in config.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigObj(value))
            else:
                setattr(self, key, value)