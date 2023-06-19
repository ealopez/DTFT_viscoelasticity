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
from scipy.fft import rfft, rfftfreq
from lib_nls import g_star,g_maxwell_model,calculate_Y
#from lmfit import minimize, Parameters


####Define some material properties to test#########
G_k=np.array([3.0e9, 1.0e8, 1.0e6])
G_e=0.0
tau_k=np.array([0.01, 0.02, 0.1 ])
####Define some material properties to test#########


# Create the time array and the Heaviside step function
P=5000
M = 100  # sequence length (non-zero values)
n = np.arange(1, P+1)
Delta_t = tau_k[0]/100.0  # timestep
t = n*Delta_t
u_n = np.heaviside(n, 1)
u_M = np.heaviside(n-M, 1)

# Calculate the relaxation modulus with noise
y_n = (u_n - u_M) * (np.sum(G_k[:, None] * np.exp(-n[None, :] * Delta_t / tau_k[:, None]), axis=0) + G_e)

# Plot the time-domain noisy data
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(t, y_n, 'bo', label='Experimental data')
#ax.plot(t, G_fit, 'r-', label='Fitted curve')
ax.set_xlabel('Time (s)')
ax.set_ylabel('G')
#ax.set_xscale('log')
#ax.set_yscale('log')
ax.legend()
plt.show()



# 2. Get RFFT of noisy time trace
rfft_values_y1 = rfft(y_n )*Delta_t  #multiplying by dt to normalize magnitudes
omega = rfftfreq(np.size(y_n), d=Delta_t)* 2.0*np.pi



# Plot the frequency-daomain noisy data
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(omega,np.abs(rfft_values_y1), color='orange',lw=3, label='g_hat_noisy_fft')
g_star_ctft = g_star(omega, G_e, G_k, tau_k)
ax.plot(omega,np.abs(g_star_ctft/(1.0j*omega)), color='yellow', lw=5, label='g_hat_ctft')

dtft = calculate_Y(omega, Delta_t, G_k, tau_k, G_e, M)
ax.plot(omega,np.abs(dtft), color='green',lw=2, label='g_hat_dtft')
#ax.plot(t, G_fit, 'r-', label='Fitted curve')
ax.set_xlabel('Omega (rad)')
ax.set_ylabel('|G|')
ax.set_xscale('log')
#ax.set_yscale('log')
ax.legend()
plt.show()




