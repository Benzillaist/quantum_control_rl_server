#%%

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]='true'
os.environ["CUDA_VISIBLE_DEVICES"]="0"
# append parent 'gkp-rl' directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
import tensorflow as tf
from tf_agents import specs
from quantum_control_rl_server import PPO
from tf_agents.networks import actor_distribution_network
from quantum_control_rl_server import remote_env_tools as rmt
from quantum_control_rl_server.h5log import h5log
import numpy as np
import pickle


sf_name = "temp_files/server_args.txt"
with open(sf_name, "rb") as fp:
    epochs, train_batch_size, init_amps, amp_scales, hdf5_name = pickle.load(fp)
fp.close()
epochs = int(epochs)
train_batch_size = int(train_batch_size)
init_amps = np.array(init_amps).astype(np.float32)
amp_scales = np.array(amp_scales).astype(np.float32)
hdf5_name = str(hdf5_name)

# print(f'server casted args: {epochs, train_batch_size, init_amps, init_freqs, amp_scales, freq_scales, hdf5_name}')

root_dir = r'C:\Users\Wang_Lab\Documents\GitLab\quantum_control_rl_server\examples\exp_1g_EXAMPLE\save_data' # root_dir = os.getcwd() #r'E:\rl_data\exp_training\pi_pulse'
host_ip = '127.0.0.1' # ip address of RL server, here it's hosted locally

num_epochs = epochs # total number of training epochs
train_batch_size = train_batch_size # number of batches to send for training epoch

do_evaluation = True # flag for implementing eval epochs or not
eval_interval = 10 # number of training epochs between eval epochs
eval_batch_size = 1 # number of batches to send for eval epoch

learn_residuals = True
save_tf_style = False

# setting up initial pulse
n_drives = int(np.shape(init_amps)[0] / 2)
n_amp_vals = np.shape(init_amps)[1] # number of array values to optimize
# print(f'n_array_vals: {n_array_vals}')

# Params for action wrapper
if len(init_amps[0]) == 1:
    action_script = {f'pulse_array_{i}': [amp_seg] for i, amp_seg in enumerate(init_amps)}
else:
    action_script = {f'pulse_array_{i}': [list(amp_seg)] for i, amp_seg in enumerate(init_amps)}


# print(f'action_script: {action_script}')

# specify shapes of actions to be consistent with the objects in action_script
# Adds amps
action_spec = {f'pulse_array_{i}': specs.TensorSpec(shape=[n_amp_vals], dtype=tf.float32)
               for i in range(2 * n_drives)}

# and for automatic differentiation of the reward
# optimal point should ideally be within +/- action_scale of the initial vals
# Adds amps
action_scale = {f'pulse_array_{i}': amp_scale for i, amp_scale in enumerate(amp_scales)}

# flags indicating whether actions will be learned or scripted
to_learn = {key: True for key in action_script.keys()}


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

print(f'root_dir: {root_dir}')
print(f'hdf5_name: {hdf5_name}')
log = h5log(root_dir, rl_params, name=hdf5_name)

############################################################
# Below code shouldn't require modification for normal use #
############################################################
# Create drivers for data collection
from quantum_control_rl_server import dynamic_episode_driver_sim_env

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
    lr = 2.5e-4,
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
    save_tf_style = save_tf_style,
    rl_params = rl_params)
