# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 10:36:22 2023

@author: quiqu

Library for non-linear least square fitting
"""

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from lib_nls import complex_modulus, NLS_fit, g_star



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
G_e = 0.0


# Define the angular frequencies
omega = np.logspace(-3, 3, num=1000)

# Calculate the complex modulus without noise
G_star_n = g_star(omega, G_e, G_k, tau_k)

# Add Gaussian noise
np.random.seed(0)  # for reproducibility
noise_level=0.05
noise = noise_level * (np.random.normal(size=G_star_n.shape) + 1j * np.random.normal(size=G_star_n.shape))
G_star_noisy = G_star_n + noise

# Save the generated data to a CSV file
df_noisy_complex = pd.DataFrame({'omega': omega, 'G_star': G_star_noisy})
#df_noisy_complex.to_csv('NoisyDataComplex.csv', index=False)






# Initialize initial guesses
Delta_t = 1.0e-3
init_params = {
    'model': complex_modulus,
    'arms': 3,
    'Ge': G_e,
    'G_i': [1.0e8, 1.0e7, 1.0e6],  # initial guesses for G1 to G5
    'tau_i': [Delta_t/10.0, Delta_t, Delta_t*10], # initial guesses for tau1 to tau5
    'delta_t':1.0e-3,
    'M':300
}

# Perform the fitting
Ge_c, tau_c, G_c = NLS_fit(omega, G_star_noisy, **init_params)


# Evaluate the complex modulus with the fitted parameters
G_complex_fit = g_star(omega, Ge_c, G_c, tau_c)


# Plot the original noisy data and the fitted data
fig, ax = plt.subplots()

# Plot original data
ax.loglog(omega, np.real(G_star_noisy), 'o', label='Original Data')

# Plot fitted data
ax.loglog(omega, np.real(G_complex_fit), 'r-', label='Fitted Data')

# Setting labels and title
ax.set_xlabel("Frequency (rad/s)")
ax.set_ylabel("G prime")
ax.set_title("Comparison between original and fitted data")

ax.legend()
plt.show()

# Plot the original noisy data and the fitted data
fig, ax = plt.subplots()

# Plot original data
ax.loglog(omega, np.imag(G_star_noisy), 'o', label='Original Data')

# Plot fitted data
ax.loglog(omega, np.imag(G_complex_fit), 'r-', label='Fitted Data')

# Setting labels and title
ax.set_xlabel("Frequency (rad/s)")
ax.set_ylabel("G biprime")
ax.set_title("Comparison between original and fitted data")

ax.legend()
plt.show()