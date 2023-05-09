#%%

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]='true'
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# append parent 'gkp-rl' directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

import tensorflow as tf
from tf_agents import specs
from quantum_control_rl_server.agents import PPO
from tf_agents.networks import actor_distribution_network
from quantum_control_rl_server.remote_env_tools import remote_env_tools as rmt
from quantum_control_rl_server.utils.h5log import h5log
import numpy as np

root_dir = os.getcwd() #r'E:\rl_data\exp_training\pi_pulse'
host_ip = '127.0.0.1' # ip address of RL server, here it's hosted locally

num_epochs = 1000 # total number of training epochs
train_batch_size = 50 # number of batches to send for training epoch

do_evaluation = True # flag for implementing eval epochs or not
eval_interval = 20 # number of training epochs between eval epochs
eval_batch_size = 1 # number of batches to send for eval epoch

learn_residuals = True
save_tf_style = False

# setting up initial pulse
n_array_vals = 40 # number of array values to optimize
t_duration = n_array_vals # 1ns resolution, since time is in ns in sim
amp = 0.5*(np.pi/t_duration) # starting 50% away from ideal pulse
init_pulse_real = list(amp*np.ones(n_array_vals)) # square real pulse
init_pulse_imag = list(np.zeros_like(init_pulse_real)) # no imag part

amp_scale = (2*np.pi)/t_duration
action_scale_array_real = list(np.ones(n_array_vals,dtype=float)*amp_scale)
action_scale_array_imag = list(np.ones(n_array_vals,dtype=float)*amp_scale)

# Params for action wrapper
action_script = {
  'pulse_array_real' : [init_pulse_real], # shape=[n_array_vals]
  'pulse_array_imag' : [init_pulse_imag]
  }

# specify shapes of actions to be consistent with the objects in action_script
action_spec = {
  'pulse_array_real' : specs.TensorSpec(shape=[n_array_vals], dtype=tf.float32),
  'pulse_array_imag' : specs.TensorSpec(shape=[n_array_vals], dtype=tf.float32)
  }

# characteristic scale of sigmoid functions used in the neural network, 
# and for automatic differentiation of the reward
# optimal point should ideally be within +/- action_scale of the initial vals
action_scale = {
  'pulse_array_real':action_scale_array_real,
  'pulse_array_imag':action_scale_array_imag
  }

# flags indicating whether actions will be learned or scripted
to_learn = {
  'pulse_array_real':True,
  'pulse_array_imag':True
  }


rl_params = {'num_epochs' : num_epochs,
             'train_batch_size' : train_batch_size,
             'do_evaluation' : do_evaluation,
             'eval_interval' : eval_interval,
             'eval_batch_size' : eval_batch_size,
             'learn_residuals' : learn_residuals,
             'action_script' : action_script,
             #'action_spec' : action_spec, # doesn't play nice with h5 files
             'action_scale': action_scale,
             'to_learn' : to_learn,
             'save_tf_style' : save_tf_style}

log = h5log(root_dir, rl_params)


############################################################
# Below code shouldn't require modification for normal use #
############################################################

# Create drivers for data collection
from quantum_control_rl_server.agents import dynamic_episode_driver_sim_env

server_socket = rmt.Server()
(host, port) = (host_ip, 5555) 
server_socket.bind((host, port))
server_socket.connect_client()

# Params for environment
env_kwargs = eval_env_kwargs = {
  'T' : 1}

# Params for reward function
reward_kwargs = {
    'reward_mode' : 'remote',
    'server_socket' : server_socket,
    'epoch_type' : 'training'}

reward_kwargs_eval = {
    'reward_mode' : 'remote',
    'server_socket' : server_socket,
    'epoch_type' : 'evaluation'}

collect_driver = dynamic_episode_driver_sim_env.DynamicEpisodeDriverSimEnv(
    env_kwargs, reward_kwargs, train_batch_size, action_script, action_scale,
    action_spec, to_learn, learn_residuals, remote=True)

eval_driver = dynamic_episode_driver_sim_env.DynamicEpisodeDriverSimEnv(
    eval_env_kwargs, reward_kwargs_eval, eval_batch_size, action_script, action_scale,
    action_spec, to_learn, learn_residuals, remote=True)

PPO.train_eval(
    root_dir = root_dir,
    random_seed = 0,
    num_epochs = num_epochs,
    # Params for train
    normalize_observations = True,
    normalize_rewards = False,
    discount_factor = 1.0,
    lr = 2.5e-3,
    lr_schedule = None,
    num_policy_updates = 20,
    initial_adaptive_kl_beta = 0.0,
    kl_cutoff_factor = 0,
    importance_ratio_clipping = 0.1,
    value_pred_loss_coef = 0.005,
    gradient_clipping = 1.0,
    entropy_regularization = 0,
    log_prob_clipping = 0.0,
    # Params for log, eval, save
    eval_interval = eval_interval,
    save_interval = 2,
    checkpoint_interval = None,
    summary_interval = 2,
    do_evaluation = do_evaluation,
    # Params for data collection
    train_batch_size = train_batch_size,
    eval_batch_size = eval_batch_size,
    collect_driver = collect_driver,
    eval_driver = eval_driver,
    replay_buffer_capacity = 15000,
    # Policy and value networks
    ActorNet = actor_distribution_network.ActorDistributionNetwork,
    zero_means_kernel_initializer = False,
    init_action_stddev = 0.08,
    actor_fc_layers = (50,20),
    value_fc_layers = (),
    use_rnn = False,
    actor_lstm_size = (12,),
    value_lstm_size = (12,),
    h5datalog = log,
    save_tf_style = save_tf_style)

# %%
