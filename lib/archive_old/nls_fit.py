# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 10:36:22 2023

@author: quiqu

Library for non-linear least square fitting
"""

import numpy as np
from lmfit import minimize, Parameters

def func_G(params, t, G_exp, arms=3):
    """Model for NLS fitting: Stress relaxation for a generalized maxwell"""
    p = params.valuesdict()
    Ge = p['Ge']
    G = np.zeros(arms)
    tau = np.zeros(arms)
    for i in range(arms):
        G[i] = p[f'G{i+1}']
        tau[i] = p[f'tau{i+1}']
    
    model = Ge + np.sum(G[:, None] * np.exp (-t/tau[:, None]) , axis=0)
    return  (model - G_exp) / G_exp  # calculating the residual



def NLS_fit_G(t, G_exp, model, arms, Ge, G_i, tau_i):
    """
    This function performs a non-linear fit based on any model given.
    """

    # create a set of Parameters
    params = Parameters()
    params.add('Ge', value=Ge, min=0)
    for i in range(arms):
        params.add(f'G{i+1}', value=G_i[i], min=0)
        params.add(f'tau{i+1}', value=tau_i[i], min=tau_i[i]/10.0, max=tau_i[i]*10.0)
    
    result = minimize(model, params, args=(t, G_exp, arms), method='leastsq')

    Ge_c = result.params['Ge'].value
    G_c= np.zeros(arms)  
    tau_c = np.zeros(arms)
    for i in range(arms):
        G_c[i] = result.params[f'G{i+1}'].value
        tau_c[i]= result.params[f'tau{i+1}'].value
    
    return Ge_c, tau_c, G_c




#import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
#from lmfit import minimize, Parameters


# Generate noisy data and perform NLS fitting

# User-defined parameters
M = 1000  # sequence length
Delta_t = 0.01  # timestep
G_e = 1  # your value of Ge
noise_level = 0.05  # adjust as necessary

# Define the directory and file
dir_path = os.getcwd()
dir_parent = os.path.dirname(dir_path)
dir = os.path.join(dir_parent, 'polyisobutylene_data')
filename = 'PIB.txt'
filepath = os.path.join(dir, filename)

# Load the CSV data
data = pd.read_csv(filepath,delimiter='\t')

# Extract G_k and tau_k arrays
G_k = data.iloc[:, 0].to_numpy()
tau_k = data.iloc[:, 1].to_numpy()

# Create the time array and the Heaviside step function
n = np.arange(1, M+1)
u_n = np.heaviside(n, 1)
u_M = np.heaviside(n-M, 1)

# Calculate the relaxation modulus with noise
#y_n = (u_n - u_M) * (np.sum(G_k[:, None] * np.exp(-n[None, :] * Delta_t / tau_k[:, None]), axis=0) + G_e)
y_n = u_n * (np.sum(G_k[:, None] * np.exp(-n[None, :] * Delta_t / tau_k[:, None]), axis=0) + G_e)

# Add Gaussian noise
np.random.seed(0)  # for reproducibility
noise = noise_level * np.random.normal(size=y_n.shape)
y_noisy = y_n + noise

# Save the generated data to a CSV file
df_noisy = pd.DataFrame({'time': n * Delta_t, 'G': y_noisy})
df_noisy.to_csv('NoisyData.csv', index=False)

# Now perform the NLS fitting on the noisy data
t = df_noisy['time'].values
G_exp = df_noisy['G'].values


init_params = {
    'model': func_G,
    'arms': 3,
    'Ge': G_e,
    'G_i': [1.0e8, 1.0e7, 1.0e6, 1.0e6, 1.0e5],  # initial guesses for G1 to G5
    'tau_i': [Delta_t/10.0, Delta_t, Delta_t*10, Delta_t*100, Delta_t*1000]  # initial guesses for tau1 to tau5
}

Ge_c, tau_c, G_c = NLS_fit_G(t, G_exp, **init_params)


# Print the results
print("Ge:", Ge_c)
print("tau:", tau_c)
print("G:", G_c)



def G_maxwell_model(t, Ge, Gk, tau_k):
    """Compute the Generalized Maxwell model."""
    return Ge + np.sum(Gk[:, None] * np.exp(-t[None, :] / tau_k[:, None]), axis=0)

# Compute the fitted curve
G_fit = G_maxwell_model(t, Ge_c, G_c, tau_c)


# Plot the experimental and fitted data
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(t, G_exp, 'bo', label='Experimental data')
ax.plot(t, G_fit, 'r-', label='Fitted curve')
ax.set_xlabel('Time (s)')
ax.set_ylabel('G')
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend()
plt.show()



