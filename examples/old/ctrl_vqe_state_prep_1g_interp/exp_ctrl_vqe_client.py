

# Author: Ben Brock 
# Created on May 02, 2023 

#%%
import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
# add quantum-control-rl dir to path for subsequent imports
#sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
import logging
import time
import pickle
from quantum_control_rl_server.remote_env_tools import Client
from exp_ctrl_vqe_cost_function import exp_g1_cost_func_xt


cf_name = "temp_files/client_args.txt"
with open(cf_name, "rb") as fp:
    client_args = pickle.load(fp)
fp.close()

# Correcting input types
# client_args[0] = np.array(client_args[0]).astype(np.float32)
client_args[0] = np.array(client_args[0]).astype(np.uint8)
client_args[6] = np.array(client_args[6]).astype(np.float32)

logger = logging.getLogger('RL')
logger.propagate = False
logger.handlers = []
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)

client_socket = Client()
(host, port) = '127.0.0.1', 5552 # ip address of RL server, here it's hosted locally
client_socket.connect((host, port))

num_drives = int(np.shape(client_args[0])[0])

# training loop
done = False
while not done:

    # receive action data from the agent (see tf_env -> reward_remote())
    message, done = client_socket.recv_data()
    logger.info('Received message from RL agent server.')
    logger.info('Time stamp: %f' %time.time())

    if done:
        logger.info('Training finished.')
        break

    # parsing message (see tf_env -> reward_remote())
    epoch_type = message['epoch_type']

    if epoch_type == 'final':
        logger.info('Final Epoch')

        locs = message['locs']
        scales = message['scales']
        for key in locs.keys():
            logger.info('locs['+str(key)+']:')
            logger.info(locs[key][0])
            logger.info('scales['+str(key)+']:')
            logger.info(scales[key][0])

        # After using the locs and scales, terminate training
        done = True
        logger.info('Training finished.')
        break

    action_batch = message['action_batch']
    batch_size = message['batch_size']
    epoch = message['epoch']

    action_batch_keys = action_batch.keys()

    # parsing action_batch and reshaping to get rid of nested structure required by tensorflow
    # here env.T=1 so the shape is (batch_size,1,pulse_len)
    new_shape_pulses = [list(action_batch[f'pulse_array_{i}'].shape) for i in range(2 * num_drives)]
    for i in range(2 * num_drives):
        new_shape_pulses[i].pop(1)
    new_shape_times = [list(action_batch[f'time_array_{i}'].shape) for i in range(2 * num_drives)]
    for i in range(2 * num_drives):
        new_shape_times[i].pop(1)

    input_pulse_arrs = np.array([action_batch[f'pulse_array_{i}'].reshape(new_shape_pulses[i]) for i in range(2 * num_drives)])
    input_time_arrs = np.array([action_batch[f'time_array_{i}'].reshape(new_shape_times[i]) for i in range(2 * num_drives)])


    logger.info('Start %s epoch %d' %(epoch_type, epoch))

    # collecting rewards for each policy in the batch
    reward_data = np.zeros((batch_size))
    for i in range(batch_size):

        # evaluating reward for ii'th element of the batch
        #   - can perform different operations depending on the epoch type
        #     for example, using more averaging for eval epochs
        if epoch_type == 'evaluation':
            reward_data[i] = exp_g1_cost_func_xt(np.append(input_pulse_arrs[:, i].flatten(), input_time_arrs[:, i].flatten()), *client_args)
        elif epoch_type == 'training':
            reward_data[i] = exp_g1_cost_func_xt(np.append(input_pulse_arrs[:, i].flatten(), input_time_arrs[:, i].flatten()), *client_args)

    # Print mean and stdev of reward for monitoring progress
    R = np.mean(reward_data)
    std_R = np.std(reward_data)
    logger.info('Average reward %.3f' %R)
    logger.info('STDev reward %.3f' %std_R)

    # send reward data back to server (see tf_env -> reward_remote())
    logger.info('Sending message to RL agent server.')
    logger.info('Time stamp: %f' %time.time())
    client_socket.send_data(reward_data)
