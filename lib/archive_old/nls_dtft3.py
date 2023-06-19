# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 10:36:22 2023

@author: quiqu

Library for non-linear least square fitting
"""

import numpy as np

#import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
#from scipy.fft import rfft, rfftfreq
from lib_nls import g_star,dtft_gt_finite, G_maxwell_finite, add_noise, zero_padding, perform_fft,g_hat, NLS_fit
from lmfit import minimize, Parameters


# Define the directory and file
dir_path = os.getcwd()
dir_parent = os.path.dirname(dir_path)
dir = os.path.join(dir_parent, 'polyisobutylene_data')
filename = 'PIB.txt'
filepath = os.path.join(dir, filename)

# Load the CSV data
data = pd.read_csv(filepath,delimiter='\t')

# Extract G_k and tau_k arrays
G_e = 0.0
G_k = data.iloc[:, 0].to_numpy()
tau_k = data.iloc[:, 1].to_numpy()



# Create the time array and the Heaviside step function
P=2**8
M = int(P/3)  # sequence length (non-zero values)
Delta_t = tau_k[-1]/100.0  # timestep
y_n,t = G_maxwell_finite(P, M, Delta_t, G_k, tau_k,G_e)
y_noisy2 = add_noise(y_n,3.0e5)
y_noisy1 = y_noisy2[0:M]
y_noisy = zero_padding(y_noisy1,P)



# Plot the time-domain noisy data
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(t, y_n, 'yo', ms=8,alpha=0.8,label='Experimental data no noise')
ax.plot(t, y_noisy, 'go', ms=5,alpha=0.5, label='Experimental with noise')
#ax.plot(t, G_fit, 'r-', label='Fitted curve')
ax.set_xlabel('Time (s)')
ax.set_ylabel('G')
#ax.set_xscale('log')
#ax.set_yscale('log')
ax.legend()
plt.show()



##############compute FFT of y_n and y_noisy#######################
rfft_y1, omega=perform_fft(y_n, Delta_t)
rfft_y1_noisy, omega=perform_fft(y_noisy, Delta_t)





# Plot the frequency-domain G prime data
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(omega,np.real(rfft_y1), color='orange',lw=3, label='g_hat_fft')
ax.plot(omega,np.real(rfft_y1_noisy), color='cyan',lw=3, label='g_hat_fft_noisy')
g_star_ctft = g_star(omega, G_e, G_k, tau_k)
ax.plot(omega,np.real(g_star_ctft/(1.0j*omega)), color='yellow', lw=5, label='g_hat_ctft')
dtft = dtft_gt_finite(omega, Delta_t, G_k, tau_k, G_e, M)
ax.plot(omega,np.real(dtft), color='green',lw=2, label='g_hat_dtft')
#ax.plot(t, G_fit, 'r-', label='Fitted curve')
ax.set_xlabel('Omega (rad)')
ax.set_ylabel('G prime')
ax.set_xscale('log')
#ax.set_yscale('log')
ax.legend()
plt.show()
a = np.real(rfft_y1)/np.real(dtft)


# Plot the frequency-domain G biprime data
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(omega,np.imag(rfft_y1), color='orange',lw=3, label='g_hat_fft')
ax.plot(omega,np.imag(rfft_y1_noisy), color='cyan',lw=3, label='g_hat_fft_noisy')
g_star_ctft = g_star(omega, G_e, G_k, tau_k)
ax.plot(omega,np.imag(g_star_ctft/(1.0j*omega)), color='yellow', lw=5, label='g_hat_ctft')
dtft = dtft_gt_finite(omega, Delta_t, G_k, tau_k, G_e, M)
ax.plot(omega,np.imag(dtft), color='green',lw=2, label='g_hat_dtft')
ax.set_xlabel('Omega (rad)')
ax.set_ylabel('G biprime')
ax.set_xscale('log')
#ax.set_yscale('log')
ax.legend()
plt.show()

b = np.imag(rfft_y1)/np.imag(dtft)

c = np.abs(rfft_y1)/np.abs(dtft)



###########NOW LET'S PERFORM THE NLS FITTING##########
# Initialize initial guesses
dt = 1.0e-3
init_params = {
    'model': g_hat,
    'arms': 3,
    'Ge': G_e,
    'G_i': [1.0e8, 1.0e7, 1.0e6],  # initial guesses for G1 to G5
    'tau_i': [dt/10.0, dt, dt*10], # initial guesses for tau1 to tau5
    'delta_t':1.0e-3,
    'M':300
}


Ge_c, tau_c, G_c = NLS_fit(omega, rfft_y1, **init_params)


# Print the results
print("Ge:", Ge_c)
print("tau:", tau_c)
print("G:", G_c)


#compute the fitted curve
g_star_dtft_fit = g_star(omega, Ge_c, G_c, tau_c)


# Plot the frequency-domain G prime data with the FIT
fig, ax = plt.subplots(figsize=(8,6))
g_star_ctft = g_star(omega, G_e, G_k, tau_k)
ax.plot(omega,np.real(g_star_ctft), color='yellow', lw=5, label='g_star_real_ctft')
ax.plot(omega,np.real(g_star_dtft_fit), color='red',lw=2, label='g_star_real_dtft')
ax.set_xlabel('omega (rad/s)')
ax.set_ylabel('G prime')
ax.set_xscale('log')
#ax.set_yscale('log')
ax.legend()
plt.show()






#############debugging G_hat##########
def g_hat2(p):
    """
    Model for non-linear optimization: Stress relaxation for a generalized maxwell model in the frequency domain
    """    
    #p = params.valuesdict()
    Delta_t = p['dt']
    G_e = p['Ge']
    arms = p['arms']
    G_k = np.zeros(arms)
    tau_k = np.zeros(arms)
    
    # Calculate frequencies for the FFT output
    omega = np.fft.fftfreq(M, d=Delta_t) * 2.0 * np.pi
        
    term1 = G_e * (1 - np.exp(-1j * M * omega * Delta_t)) / (1 - np.exp(-1j * omega * Delta_t))
    
    term2 = np.sum([G_k[k] * (1 - np.exp(-M * Delta_t / tau_k[k]) * np.exp(-1j * M * omega * Delta_t)) / (1 - np.exp(-Delta_t / tau_k[k]) * np.exp(-1j * omega * Delta_t)) for k in range(arms)], axis=0)
    
    #model = Delta_t * (term1 + term2)
    return (term1+term2)



params_ini = {
    #'model': g_hat,
    'arms': len(tau_k),
    'Ge': G_e,
    'G_i': G_k,  # initial guesses for G1 to G5
    'tau_i': tau_k, # initial guesses for tau1 to tau5
    'dt':Delta_t,
    'M':M
}

#params_ini=Parameters()  #intermediately needed

dtft = g_hat2(params_ini)