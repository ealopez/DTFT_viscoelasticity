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
P = 2**12  # sequence length including zero padded values
M = int(P/10) #non-zero values of the relaxation modulus
Delta_t = tau_k[3]/100.0  # timestep
G_e = 0.0  #equlibrium modulus, for PIB assumed zero

# Define the directory and file for material properties
mylibdir = os.path.join(os.path.dirname(os.getcwd()), 'lib')
os.chdir(mylibdir)

from lib_nls import g_star,func_g,zero_padding,add_noise,dtft_gt_finite,perform_fft,g_hat,nls_fit,g_maxwell_finite,calculate_g_nu,g_hat_ctft

#Simulate relaxation modulus with and without noise
y_n,tn= g_maxwell_finite(P,M,Delta_t,G_k,tau_k,G_e)

g_nu,omega = perform_fft(y_n, Delta_t)  #numerical FFT of input

dtft,_ = dtft_gt_finite(omega[1::], Delta_t, G_k, tau_k, G_e, M)  #theoretical DTFT

fig, ax = plt.subplots(figsize=(8,6))
ax.plot(omega[1::],g_nu[1::].real, color='orange',lw=8, label='g_nu_fft')
ax.plot(omega[1::],dtft.real, color='cyan',lw=5, label='g_rel_dtft_an')
#ax.plot(omega[1::],dtft_fit.real, color='red',lw=3, label='g_rel_dtft_fit')
#ax.plot(omega,np.real(dtft2), color='yellow',lw=3, label='g_nu_dtft2')
ax.set_xlabel('Omega (rad/s)')
ax.set_ylabel('G_nu')
ax.set_xscale('log')
#ax.set_yscale('log')
ax.legend()
plt.show()






#grel,t = g_maxwell_finite(P, M, Delta_t, G_k, tau_k)
#g_nu,omega = perform_fft(grel, Delta_t)

y_n2,tn2 = g_maxwell_finite(M,M,Delta_t,G_k,tau_k,G_e)  #relaxation modulus without zero padding
noise_level = 1.0e8 #1.0e4  # adjust as necessary
y_noise = add_noise(y_n2,noise_level) #noise is added for the simulation trace (added to the non-padded version)
y_noisy = zero_padding(y_noise,P)  #Stress-relaxation for PIB: simulated trace with Gaussian noise

ini_par = {
    'model': func_g,
    'arms': 3,
    'Ge': 0.0,
    'G_i': [5.0e9,1.0e1,1.0e8],  # initial guesses for G1 to G3
    'tau_i': [Delta_t/10.0,Delta_t,Delta_t*10], # initial guesses for tau1 to tau3
    'delta_t':Delta_t,
    'M':M
}

Ge_grel_c, tau_grel_c, G_grel_c, res = nls_fit(tn,y_noisy,**ini_par)

#evaluate Grel for fitted parameters
grel_fit, tn_fit =g_maxwell_finite(P,M,Delta_t,G_grel_c,tau_grel_c,Ge_grel_c)

# Plot the relaxation modulus with and without noise
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(tn, y_n, label='Theoretical',color='yellow',lw=5)
ax.plot(tn, y_noisy,'o', label='With Noise', color='green',lw=3)
ax.plot(tn_fit,grel_fit, label='NLS fit', color='red',linestyle='dashed',lw=3)
ax.set_xscale('log')
#ax.set_yscale('log')
ax.set_xlabel('Time')
ax.set_ylabel('Relaxation Modulus')
ax.legend()
plt.show()

g_nu,omega = perform_fft(y_noisy, Delta_t)  #numerical FFT of input

dtft,_ = dtft_gt_finite(omega[1::], Delta_t, G_k, tau_k, G_e, M)  #theoretical DTFT

#dtft_orig = dtft_gt_finite(omega[1::], Delta_t, G_k, tau_k, G_e, M)

ctft = calculate_g_nu(G_e, G_k, tau_k, omega) #theoretical CTFT

ini_par_dtft = {
    'model': g_hat,
    'arms': 3,
    'Ge': 0.0,
    'G_i': [5.0e9,1.0e1,1.0e8],  # initial guesses for G1 to G3
    'tau_i': [Delta_t/10.0,Delta_t,Delta_t*10], # initial guesses for tau1 to tau3
    'delta_t':Delta_t,
    'M':M
}


Ge_dtft_fit, tau_dtft_fit, G_dtft_fit, res_dtft= nls_fit(omega,g_nu, **ini_par_dtft)

ini_par_ctft = {
    'model': g_hat_ctft,
    'arms': 3,
    'Ge': 0.0,
    'G_i': [5.0e9,1.0e1,1.0e8],  # initial guesses for G1 to G3
    'tau_i': [Delta_t/10.0,Delta_t,Delta_t*10], # initial guesses for tau1 to tau3
    'delta_t':Delta_t,
    'M':M
}

Ge_ctft_fit, tau_ctft_fit, G_ctft_fit, res_ctft= nls_fit(omega[1:-1],g_nu[1:-1], **ini_par_ctft)

dtft_fit,_ = dtft_gt_finite(omega[1:-1], Delta_t, G_dtft_fit, tau_dtft_fit, Ge_dtft_fit, M)  #theoretical DTFT

#plot Fourier domain functions and fits
fig, ax = plt.subplots(figsize=(10, 6))
#ax.plot(omega[1::],np.abs(ctft[1::]), label='CTFT', color='yellow',lw=5)
ax.plot(omega[1::],np.real(g_nu[1::]), label='FFT',color='blue',lw=10,alpha=0.3)
ax.plot(omega[1::],np.real(dtft), label='DTFT', color='green',lw=3)
ax.plot(omega[1:-1],np.real(dtft_fit), label='NLS fit', color='red',linestyle='dashed',lw=3)
ax.set_xscale('log')
#ax.set_yscale('log')
ax.set_xlabel('omega (rad/s)')
ax.set_ylabel('g_nu real')
ax.legend()
plt.show()


gstar_theory = g_star(omega[1::], G_e, G_k, tau_k)
gstar_dtft = g_star(omega[1::], Ge_dtft_fit, G_dtft_fit, tau_dtft_fit)
gstar_ctft = g_star(omega[1::], Ge_ctft_fit, G_ctft_fit, tau_ctft_fit)
gstar_grel = g_star(omega[1::], Ge_grel_c, G_grel_c, tau_grel_c)

#plot complex modulus
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(omega[1::],np.real(gstar_theory), label='Theory', color='yellow',lw=5)
ax.plot(omega[1::],np.real(gstar_dtft), label='DTFT_fit',color='blue',lw=3)
ax.plot(omega[1::],np.real(gstar_ctft), label='CTFT fit', color='green',lw=3)
ax.plot(omega[1::], np.real(gstar_grel), label='Grel fit', color='red',linestyle='dashed',lw=3)
ax.set_xscale('log')
#ax.set_yscale('log')
ax.set_ylim(min(np.abs(gstar_theory))*0.9,max(np.abs(gstar_theory))*1.1)
ax.set_xlabel('omega (rad/s)')
ax.set_ylabel('|G|')
ax.legend()
plt.show()