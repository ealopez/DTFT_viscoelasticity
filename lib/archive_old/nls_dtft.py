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
from lib_nls import g_star,X,G_maxwell_model
#from lmfit import minimize, Parameters


# Generate noisy data and perform NLS fitting

# User-defined parameters
P=5000
M = 50  # sequence length
n = np.arange(1,P+1)
Delta_t = 0.001  # timestep
t = n*Delta_t
noise_level = 0.0  # adjust as necessary

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
n = np.arange(1, P+1)
u_n = np.heaviside(n, 1)
u_M = np.heaviside(n-M, 1)


####Define some material properties to test#########
G_k=np.array([3.0e9, 1.0e8, 1.0e6])
Ge=0.0
tau_k=np.array([0.01, 0.02, 0.1 ])


sf = 100.0/tau_k[0]
Delta_t = 1.0/sf
t = n*Delta_t
####Define some material properties to test#########

# Calculate the relaxation modulus with noise
y_n = (u_n - u_M) * (np.sum(G_k[:, None] * np.exp(-n[None, :] * Delta_t / tau_k[:, None]), axis=0) + G_e)
#y_n = u_n * (np.sum(G_k[:, None] * np.exp(-n[None, :] * Delta_t / tau_k[:, None]), axis=0) + G_e)




#y_n = G_maxwell_model(t, G_k, tau_k,0.0)

# Add Gaussian noise
np.random.seed(0)  # for reproducibility
noise = noise_level * np.random.normal(size=y_n.shape)
y_noisy = y_n + noise

# 2. Get RFFT of noisy time trace
rfft_values_y1 = rfft(y_noisy )*Delta_t  #multiplying by dt to normalize magnitudes
omega = rfftfreq(np.size(y_noisy), d=Delta_t)* 2.0*np.pi



# Save the generated data to a CSV file
df_noisy = pd.DataFrame({'time': n * Delta_t, 'G': y_noisy})
#df_noisy.to_csv('NoisyData.csv', index=False)




# Plot the time-domain noisy data
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(t, y_noisy, 'bo', label='Experimental data')
#ax.plot(t, G_fit, 'r-', label='Fitted curve')
ax.set_xlabel('Time (s)')
ax.set_ylabel('G')
#ax.set_xscale('log')
#ax.set_yscale('log')
ax.legend()
plt.show()

# Plot the frequency-daomain noisy data
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(omega,np.abs(rfft_values_y1), color='orange',lw=3, label='g_hat_noisy_fft')
g_star_ctft = g_star(omega, G_e, G_k, tau_k)
ax.plot(omega,np.abs(g_star_ctft/(1.0j*omega)), color='yellow', lw=5, label='g_hat_ctft')

def calculate_Y(omega, Delta_t, G_k, tau_k, G_e, M):
    term1 = sum( [ G_e * (1.0 - np.exp(-M * Delta_t / tau_k[k])*np.exp(1.0j * omega * Delta_t)) / (1.0 - np.exp(1.0j * omega * Delta_t)) for k in range(len(tau_k))])

    term2 = sum ([G_k[k] * (1 - np.exp(-M * Delta_t / tau_k[k]) * np.exp(1.0j * omega * Delta_t)) / (1 - np.exp(-Delta_t / tau_k[k]) * np.exp(1.0j * omega * Delta_t)) for k in range(len(G_k))] )

    Y = Delta_t * (term1 + term2)
    return Y



dtft = calculate_Y(omega, Delta_t, G_k, tau_k, G_e, M)
ax.plot(omega,np.abs(dtft), color='green',lw=2, label='g_hat_dtft')
#ax.plot(t, G_fit, 'r-', label='Fitted curve')
ax.set_xlabel('Omega (rad)')
ax.set_ylabel('|G|')
ax.set_xscale('log')
#ax.set_yscale('log')
ax.legend()
plt.show()




