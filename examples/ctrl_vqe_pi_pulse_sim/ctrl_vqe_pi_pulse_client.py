

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
from utils.opt_utils import ti_cost_eval


def cost_q_e(final_expect, final_dm):
    return(final_expect[0])

def cost_qA_g1(final_expect, final_dm):
    return np.abs(final_dm.full()[1][0])

def cost_qAB_g11(final_expect, final_dm):
    return np.abs(final_dm.full()[7][0])

def cost_qAB_g11_n(final_expect, final_dm):
    noise = (np.random.rand(1)[0] * 0.04) - 0.02
    return np.abs(final_dm.full()[7][0]) + noise

cf_name = "temp_files/client_args.txt"
with open(cf_name, "rb") as fp:
    client_args = pickle.load(fp)
fp.close()

client_args[0] = int(client_args[0])
client_args[1] = int(client_args[1])
# client_args[2] = np.array(client_args[2]).astype(qt.Qobj)
client_args[3] = np.array(client_args[3]).astype(np.float32)
client_args[4] = float(client_args[4])
# client_args[7] = np.array(client_args[7]).astype(qt.Qobj)
# client_args[8] = np.array(client_args[8]).astype(qt.Qobj)
# client_args[10] = np.array(client_args[10]).astype(qt.Qobj)
client_args[11] = np.array(client_args[11]).astype(np.float32)

# client_args = [num_drives, num_elems, drive_ops, t_segs, t_step, H_0, init_state, c_ops, eval_ops, sim_options, element_ops, element_freqs, output_cost_func, verbose]

# print(f'casted client_args: {client_args}')

logger = logging.getLogger('RL')
logger.propagate = False
logger.handlers = []
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)

client_socket = Client()
(host, port) = '127.0.0.1', 5555 # ip address of RL server, here it's hosted locally
client_socket.connect((host, port))

num_drives = client_args[0]

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
    new_shape_freqs = [list(action_batch[f'freq_{i}'].shape) for i in range(num_drives)]
    for i in range(num_drives):
        new_shape_freqs[i].pop(1)
    # new_shape_list_real = list(action_batch['pulse_array_real'].shape)
    # new_shape_list_imag = list(action_batch['pulse_array_imag'].shape)
    # new_shape_list_real.pop(1)
    # new_shape_list_imag.pop(1)
    input_pulse_arrs = np.array([action_batch[f'pulse_array_{i}'].reshape(new_shape_pulses[i]) for i in range(2 * num_drives)])
    input_freq_arrs = np.array([action_batch[f'freq_{i}'].reshape(new_shape_freqs[i]) for i in range(num_drives)])

    # amp_segs = np.array([action_batch[f'pulse_array_{i}'].reshape(new_shape_amps[i]) for i in range(len(new_shape_amps))]).flatten()
    # amp_segs = np.array([action_batch[f'freqs_{i}'].reshape(new_shape_freqs[i]) for i in range(len(new_shape_amps))]).flatten()
    # freqs = np.array([action_batch[f'pulse_freqs']].reshape(new_shape_freqs))
    # real_pulses = action_batch['pulse_array_real'].reshape(new_shape_list_real)
    # imag_pulses = action_batch['pulse_array_imag'].reshape(new_shape_list_imag)
    # for i in range(len(amp_segs)):
    #     print(f'pulse amps {i}: pulse_amps{i}')
    # print(f'real_pulses: {real_pulses}')
    # print(f'imag_pulses: {imag_pulses}')

    # input_arrs = np.array(input_pulse_arrs + input_freq_arrs)


    logger.info('Start %s epoch %d' %(epoch_type, epoch))

    # print(f'input_arrs: {input_arrs}')
    # print(f'input_pulse_arrs: {input_pulse_arrs}')
    # print(f'input_freq_arrs: {input_freq_arrs}')

    # collecting rewards for each policy in the batch
    reward_data = np.zeros((batch_size))
    for i in range(batch_size):

        # print(f'input_pulse_arrs: {input_pulse_arrs[:, i]}')
        # print(f'input_freq_arrs: {input_freq_arrs[:, i]}')

        # print(f'input_arrs: {np.append(input_pulse_arrs[:, i].flatten(), input_freq_arrs[:, i])}')

        # evaluating reward for ii'th element of the batch
        #   - can perform different operations depending on the epoch type
        #     for example, using more averaging for eval epochs
        if epoch_type == 'evaluation':
            reward_data[i] = ti_cost_eval(np.append(input_pulse_arrs[:, i].flatten(), input_freq_arrs[:, i]), *client_args)
        elif epoch_type == 'training':
            reward_data[i] = ti_cost_eval(np.append(input_pulse_arrs[:, i].flatten(), input_freq_arrs[:, i]), *client_args)

    # Print mean and stdev of reward for monitoring progress
    R = np.mean(reward_data)
    std_R = np.std(reward_data)
    logger.info('Average reward %.3f' %R)
    logger.info('STDev reward %.3f' %std_R)

    # send reward data back to server (see tf_env -> reward_remote())
    logger.info('Sending message to RL agent server.')
    logger.info('Time stamp: %f' %time.time())
    client_socket.send_data(reward_data)
