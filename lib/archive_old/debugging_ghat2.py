# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 10:36:22 2023

@author: quiqu

Library for non-linear least square fitting
"""

import numpy as np

#import numpy as np
import pandas as pd
import os
from lib_nls import dtft_gt_finite,perform_fft,g_hat,nls_fit,g_maxwell_finite,calculate_g_nu,g_hat_ctft
from scipy.fft import rfftfreq
import matplotlib.pyplot as plt


# Define the directory and file
dir_path = os.getcwd()
dir_parent = os.path.dirname(dir_path)
mydir = os.path.join(dir_parent, 'polyisobutylene_data')
filename = 'PIB.txt'
filepath = os.path.join(mydir, filename)

# Load the CSV data
data = pd.read_csv(filepath,delimiter='\t')

"""
# Extract G_k and tau_k arrays
G_e = 0.0
G_k = data.iloc[:, 0].to_numpy()
tau_k = data.iloc[:, 1].to_numpy()


# Create the time array and the Heaviside step function
P=2**12
M = int(P/3)  # sequence length (non-zero values)
Delta_t = tau_k[-1]/100.0  # timestep
"""

# Extract G_k and tau_k arrays
G_e = 0.0
G_k = data.iloc[:, 1].to_numpy()
tau_k = data.iloc[:, 0].to_numpy()


# Create the time array and the Heaviside step function
P=2**12
M = int(P/3)  # sequence length (non-zero values)
Delta_t = tau_k[5]/100.0  # timestep





omega = rfftfreq(P,d=Delta_t)* 2.0 * np.pi
dtft2 = dtft_gt_finite(omega, Delta_t, G_k, tau_k, G_e, M)
gnu_ctft = calculate_g_nu(G_e, G_k, tau_k, omega)


fig, ax = plt.subplots(figsize=(8,6))
ax.plot(omega,np.real(dtft2), color='cyan',lw=5, label='g_hat_dtft')
ax.plot(omega,np.real(gnu_ctft), color='green',lw=3, label='g_hat_ctft')
ax.set_xlabel('Omega (rad)')
ax.set_ylabel('G prime')
ax.set_xscale('log')
#ax.set_yscale('log')
ax.legend()
plt.show()




#################testing NLS for the Fourier-domain relaxation modulus using CTFT#############
ini_par = {
    'model': g_hat_ctft,
    'arms': 3,
    'Ge': 0.0,
    'G_i': [1.0e9,1.0e8,1.0e7],  # initial guesses for G1 to G3
    'tau_i': [tau_k[-1]/10.0,tau_k[-1],tau_k[-1]*10], # initial guesses for tau1 to tau3
    'delta_t':Delta_t,
    'M':M
}


Ge_fit1, tau_fit1, G_fit1= nls_fit(omega,gnu_ctft, **ini_par)
gnu_ctft_fit = calculate_g_nu(Ge_fit1, G_fit1, tau_fit1, omega)

fig, ax = plt.subplots(figsize=(8,6))
ax.plot(omega,gnu_ctft.real, color='orange',lw=8, label='g_nu_ctft')
ax.plot(omega,gnu_ctft_fit.real, color='red',lw=5, label='g_nu_ctft_fit')
ax.set_xlabel('Omega (rad/s)')
ax.set_ylabel('G_nu')
ax.set_xscale('log')
#ax.set_yscale('log')
ax.legend()
plt.show()



#################testing NLS for the Fourier-domain relaxation modulus using DTFT#############
grel,t = g_maxwell_finite(P, M, Delta_t, G_k, tau_k)
g_nu,omega = perform_fft(grel, Delta_t)

ini_par = {
    'model': g_hat,
    'arms': 3,
    'Ge': 0.0,
    'G_i': [1.0e9,1.0e8,1.0e7],  # initial guesses for G1 to G3
    'tau_i': [tau_k[-1]/10.0,tau_k[-1],tau_k[-1]*10], # initial guesses for tau1 to tau3
    'delta_t':Delta_t,
    'M':M
}


Ge_fit, tau_fit, G_fit= nls_fit(omega[1::],g_nu[1::], **ini_par)

dtft_orig = dtft_gt_finite(omega[1::], Delta_t, G_k, tau_k, G_e, M)

#Ge_fit, tau_fit, G_fit= nls_fit(omega[1::],dtft_orig, **ini_par)

dtft_fit = dtft_gt_finite(omega[1::], Delta_t, G_fit, tau_fit, Ge_fit, M)

dtft2 = dtft_gt_finite(omega, Delta_t, G_k, tau_k, G_e, M)

fig, ax = plt.subplots(figsize=(8,6))
ax.plot(omega[1::],g_nu[1::].real, color='orange',lw=8, label='g_nu_fft')
ax.plot(omega[1::],dtft_orig.real, color='cyan',lw=5, label='g_rel_dtft_an')
ax.plot(omega[1::],dtft_fit.real, color='red',lw=3, label='g_rel_dtft_fit')
ax.plot(omega,np.real(dtft2), color='yellow',lw=3, label='g_nu_dtft2')
ax.set_xlabel('Omega (rad/s)')
ax.set_ylabel('G_nu')
ax.set_xscale('log')
#ax.set_yscale('log')
ax.legend()
plt.show()


