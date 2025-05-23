import multiprocessing as mp
import time
import os

def start_server():
    os.system('start /wait cmd /k C:\\Users\\Wang_Lab\\anaconda3\\Scripts\\activate.bat C:\\Users\\Wang_Lab\\anaconda3 ^&^& cd C:\\Users\\Wang_Lab\Documents\\GitLab\\quantum_control_rl_server\\examples\\exp_1g_interp_amp ^&^& conda activate qcrl-server ^&^& python ctrl_vqe_pi_pulse_training_server.py')

def start_client():
    os.system('start /wait cmd /k C:\\Users\\Wang_Lab\\anaconda3\\Scripts\\activate.bat C:\\Users\\Wang_Lab\\anaconda3 ^&^& cd C:\\Users\\Wang_Lab\Documents\\GitLab\\quantum_control_rl_server\\examples\\exp_1g_interp_amp ^&^& conda activate qcrl-server ^&^& python ctrl_vqe_pi_pulse_client.py')

if __name__ == '__main__':
    ps = mp.Process(name="ps", target=start_server)
    pc = mp.Process(name="pc", target=start_client)
    ps.start()
    time.sleep(10)
    pc.start()