

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

from quantum_control_rl_server.remote_env_tools import Client

from pi_pulse_sim_function import pi_pulse_sim

logger = logging.getLogger('RL')
logger.propagate = False
logger.handlers = []
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)


client_socket = Client()
(host, port) = '127.0.0.1', 5555 # ip address of RL server, here it's hosted locally
client_socket.connect((host, port))

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

    # parsing action_batch and reshaping to get rid of nested
    # structure [[float_param]] required by tensorflow
    amplitudes = action_batch['amp'].reshape([batch_size])
    drags = action_batch['drag'].reshape([batch_size])
    detunings = action_batch['detuning'].reshape([batch_size])

    logger.info('Start %s epoch %d' %(epoch_type, epoch))

    # collecting rewards for each policy in the batch
    reward_data = np.zeros((batch_size))
    for ii in range(batch_size):

        # evaluating reward for ii'th element of the batch
        #   - can perform different operations depending on the epoch type
        #     for example, using more averaging for eval epochs
        if epoch_type == 'evaluation':
            reward_data[ii] = pi_pulse_sim(amplitudes[ii],drags[ii],detunings[ii])
        elif epoch_type == 'training':
            reward_data[ii] = pi_pulse_sim(amplitudes[ii],drags[ii],detunings[ii])
        # elif epoch_type == 'final':
        #     print('Got to final epoch!')
        #     print('amplitudes:')
        #     print(amplitudes)
        #     print('drags:')
        #     print(drags)
        #     print('detunings:')
        #     print(detunings)
        #     done = True
        #     logger.info('Training finished.')
        #     break
        
    # Print mean and stdev of reward for monitoring progress
    R = np.mean(reward_data) 
    std_R = np.std(reward_data)
    logger.info('Average reward %.3f' %R)
    logger.info('STDev reward %.3f' %std_R)
    
    # send reward data back to server (see tf_env -> reward_remote())
    logger.info('Sending message to RL agent server.')
    logger.info('Time stamp: %f' %time.time())
    client_socket.send_data(reward_data)
    


# %%
