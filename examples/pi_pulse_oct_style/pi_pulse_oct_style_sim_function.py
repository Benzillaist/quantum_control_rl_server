# Author: Ben Brock 
# Created on May 03, 2023 

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

def pi_pulse_oct_style_sim(pulse_array_real, # real pulse
                            pulse_array_imag, # imag pulse
                            N = 5, # number of transmon states
                            kerr = -0.1, # kerr nonlinearity of transmon in GHz
                            n_times = 101 # number of time-steps for the qutip simulation
                            ):

    q = qt.destroy(N)
    psi0 = qt.fock(N,0)

    
    # frequencies in GHz, times in ns
    t_duration = len(pulse_array_real)
    ts = np.linspace(-t_duration/2,t_duration/2,n_times)

    H0 = 2*np.pi*kerr*(q.dag()**2)*(q**2)
    H1 = q.dag()
    H2 = q

    pulse = pulse_array_real + 1j*pulse_array_imag
    print(f'pulse: {pulse}')
    pulse_func = qt.interpolate.Cubic_Spline(ts[0], ts[-1], pulse)
    print(f'pulse_func: {pulse_func}')
    pulse_conj_func = qt.interpolate.Cubic_Spline(ts[0], ts[-1], pulse.conj())
    print(f'pulse_conj_func: {pulse_conj_func}')

    t_arr = np.linspace(-3, 3, 101)
    plt.plot(t_arr, pulse_func(t_arr).real, label="0I")
    # plt.plot(t_arr, pulse_func(t_arr).imag, label="0Q")
    plt.plot(t_arr, pulse_conj_func(t_arr).real, label="1I")
    # plt.plot(t_arr, pulse_conj_func(t_arr).imag, label="1Q")
    plt.legend()
    plt.show()

    H = [H0,[H1,pulse_func],[H2,pulse_conj_func]]
    result = qt.sesolve(H,psi0,tlist=ts)
    this_reward = (2*qt.expect(qt.fock_dm(N,1),result.states[-1]))-1 # return reward as pauli measurement
    return this_reward

if __name__ == '__main__':
    pi_pulse_oct_style_sim(np.array([1, 2, 3]), np.array([4, 5, 6]), N=10, kerr=0)