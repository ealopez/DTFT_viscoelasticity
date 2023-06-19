# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 10:36:22 2023

@author: quiqu

Library for non-linear least square fitting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Define the directory and file for material properties
mydir = os.path.join(os.path.dirname(os.getcwd()), 'polyisobutylene_data')
filename = 'PIB.txt'
filepath = os.path.join(mydir, filename)

# Load the CSV data
data = pd.read_csv(filepath,delimiter='\t')

# Extract G_k and tau_k arrays
tau_k = data.iloc[:, 0].to_numpy()
G_k = data.iloc[:, 1].to_numpy()

# Creating the DataFrame
df = pd.DataFrame({'tau_k': tau_k,'G_k': G_k })

# Sort the DataFrame by tau_k in ascending order
pib = df.sort_values(by='tau_k')

# Print sorted DataFrame
#print(pib)

# User-defined parameters
P = 2**12 # sequence length including zero padded values
M = int(P/15) #non-zero values of the relaxation modulus
Delta_t = tau_k[5]*1000.0  # timestep
G_e = 0.0  #equlibrium modulus, for PIB assumed zero

"""
n = np.arange(0, P)  #or from 1, P+1
t=n*Delta_t
u_n = np.heaviside(n, 1)
u_M = np.heaviside(n-M+1, 0)
"""

# Define the directory and file for material properties
mylibdir = os.path.join(os.path.dirname(os.getcwd()), 'lib')
os.chdir(mylibdir)


from lib_nls import func_g,zero_padding,add_noise,dtft_gt_finite,perform_fft,g_hat,nls_fit,g_maxwell_finite,calculate_g_nu,g_hat_ctft


#Simulate relaxation modulus with and without noise
y_n,tn= g_maxwell_finite(P,M,Delta_t,G_k,tau_k,G_e)



def relaxation_modulus(delta_t, M, P, Ge, Gk, tau_k):
    """
    Function to calculate the relaxation modulus.
    
    Parameters:
    delta_t : float
        Time step
    M : int
        Length of the non-zero part of the time array
    P : int
        Total length of the output array
    Ge : float
        Modulus
    Gk : numpy array
        Array of moduli
    tau_k : numpy array
        Array of relaxation times

    Returns:
    G : numpy array
        The relaxation modulus at time points `t`
    """
    
    # Ensure Gk and tau_k are numpy arrays
    Gk = np.array(Gk)
    tau_k = np.array(tau_k)
    
    # Compute the time points
    t = np.arange(P) * delta_t

    # Initialize G as an array of zeros of length P
    G = np.zeros(P)
    
    # Compute the relaxation modulus for the first M time points
    G[:M] = Ge + np.sum(Gk[:, None] * np.exp(-t[:M] / tau_k[:, None]), axis=0)

    return G,t

"""
y_n,tn = relaxation_modulus(Delta_t, M, P, G_e, G_k, tau_k)
"""

g_nu,omega = perform_fft(y_n, Delta_t)  #numerical FFT of input

dtft = dtft_gt_finite(omega[1::], Delta_t, G_k, tau_k, G_e, M)  #theoretical DTFT

fig, ax = plt.subplots(figsize=(8,6))
ax.plot(omega[1::],g_nu[1::].real, color='orange',lw=8, label='g_nu_fft')
ax.plot(omega[1::],dtft.real, color='cyan',lw=3, label='g_rel_dtft_an')
#ax.plot(omega[1::],dtft_fit.real, color='red',lw=3, label='g_rel_dtft_fit')
#ax.plot(omega,np.real(dtft2), color='yellow',lw=3, label='g_nu_dtft2')
ax.set_xlabel('Omega (rad/s)')
ax.set_ylabel('G_nu')
ax.set_xscale('log')
#ax.set_yscale('log')
ax.legend()
plt.show()




