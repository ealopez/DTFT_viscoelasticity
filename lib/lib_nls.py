# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 10:36:22 2023

@author: quiqu

Library for non-linear least square fitting
"""

import numpy as np
from lmfit import minimize, Parameters
from scipy.fft import rfft, rfftfreq



def calculate_g_nu(G_e, G_k, tau_k, omega):
    """
    Function to calculate G_nu in the context of Coninuous-time fourier transform.
    
    Parameters:
    G_e (float): Value of G_e.
    G_k (ndarray): Array of G_k values.
    tau_k (ndarray): Array of tau_k values.
    omega (float): Value of omega.
    
    Returns:
    float: Calculated value of G_nu.
    """
    
    G_nu = G_e / (1j * omega) +np.sum([G_k[i] * tau_k[i] / (1 + 1j * omega * tau_k[i]) for i in range(len(G_k))],axis=0)
          
    return G_nu


def dtft_gt_finite(omega, delta_t, G, tau, Ge, M):
    """
    function that calculates the DTFT of a discrete and finite sequence

    Parameters
    ----------
    omega : TYPE
        DESCRIPTION.
    Delta_t : TYPE
        DESCRIPTION.
    G_k : TYPE
        DESCRIPTION.
    tau_k : TYPE
        DESCRIPTION.
    G_e : TYPE
        DESCRIPTION.
    M : TYPE
        DESCRIPTION.

    Returns
    -------
    Y : TYPE
        DESCRIPTION.

    """
    first_term = sum([G[i] / (1 - np.exp(-delta_t / tau[i]) * np.exp(-1j * omega * delta_t)) for i in range(len(G))])
    second_term = sum([G[i] * np.exp(-M * delta_t / tau[i]) * np.exp(-1j * M * omega * delta_t) / (1 - np.exp(-delta_t / tau[i]) * np.exp(-1j * omega * delta_t)) for i in range(len(G))])
    third_term = Ge / (1 - np.exp(-1j * omega * delta_t))
    fourth_term = Ge * np.exp(-1j * M * omega * delta_t) / (1 - np.exp(-1j * omega * delta_t))
    aliased_only = ( first_term+third_term)*delta_t
    aliased_leaked = (first_term - second_term + third_term - fourth_term)*delta_t
    return aliased_leaked, aliased_only




def perform_fft(y_n, Delta_t):
    """
    Function that performs a Fast Fourier Transform (FFT) on an input array, multiplies it by the time step to normalize
    the magnitudes, and calculates the corresponding frequencies. The function only returns the positive half of the 
    spectrum and the corresponding frequencies.

    Parameters
    ----------
    y_n : np.array
        The input array to perform the FFT on.
    Delta_t : float
        The time step that was used for sampling the input data. This is used to normalize the magnitudes of the FFT.

    Returns
    -------
    rfft_values_y1 : np.array
        The positive half of the FFT of the input array, with magnitudes normalized by Delta_t.
    omega : np.array
        The corresponding frequencies for the positive half of the FFT.
    """
    # Perform FFT and normalize magnitudes
    rfft_values_y1 = rfft(y_n) * Delta_t

    # Calculate frequencies for the FFT output
    omega = rfftfreq(len(y_n),d=Delta_t)* 2.0 * np.pi

    return rfft_values_y1, omega



def zero_padding(y_n, P):
    """
    Function that pads an array with zeros at the end until its size equals a given value.

    Parameters
    ----------
    y_n : np.array
        The original array to be padded.
    P : int
        The desired total size of the array after padding.

    Returns
    -------
    np.array
        The padded array. If P > len(y_n), the returned array has a size of P with the additional elements filled with zeros.
        If P <= len(y_n), the original array y_n is returned without any changes.
    """
    if P > len(y_n):
        return np.pad(y_n, (0, P - len(y_n)), 'constant')
    else:
        return y_n



def add_noise(y_n,noise_level):
    """
    function that adds Gaussian noise to time trace y_n input

    Parameters
    ----------
    noise_level : float
        level of Gaussian noise
    y_n : np.array
        original array without noise

    Returns
    -------
    y_noisy : np.array
        returned array with Gaussian noise added

    """
    # Add Gaussian noise
    np.random.seed(0)  # for reproducibility
    noise = noise_level * np.random.normal(size=y_n.shape)
    y_noisy = y_n + noise
    return y_noisy

def g_maxwell_finite(P,M,Delta_t,G_k,tau_k,G_e=0.0):
    """
    Relaxation modulus (stress relaxation) for a Generalized Maxwell for a truncated function

    Parameters
    ----------
    P : int
        length of the total array including zeros padded
    M : int
        length of array inlcuding only non-zero values
    Delta_t : float
        experimental timestep (inverse of sampling frequency)
    G_k : np.array
        array of values of moduli of the Maxwell arms
    tau_k : np.array
        array of values of relaxation times
    Ge : float, optional
        Equlibrium (rubbery) modulus. The default is 0.0.

    Returns
    -------
    y_n : np.array
        array of values of relaxation modulus for the discrete-finite version

    """
    # Create the time array and the Heaviside step function
    n = np.arange(0, P)  #or from 1, P+1
    t=n*Delta_t
    u_n = np.heaviside(n, 1)
    u_M = np.heaviside(n-M+1, 0)  #or n-M,0

    # Calculate the relaxation modulus
    y_n = (u_n - u_M) * (np.sum(G_k[:, None] * np.exp(-n[None, :] * Delta_t / tau_k[:, None]), axis=0) + G_e)
    return y_n,t


def g_maxwell_model(t, G_k, tau_k,Ge=0.0):
    """Compute the Generalized Maxwell model."""
    return Ge + np.sum(G_k[:, None] * np.exp(-t[None, :] / tau_k[:, None]), axis=0)


def g_star(omega, Ge, G, tau):
    """
    Calculates the complex modulus for a generalized Maxwell model.

    Parameters:
    omega (np.array): Array of radian frequencies.
    Ge (float): Rubbery modulus.
    G (np.array): Array of moduli for each Maxwell arm.
    tau (np.array): Array of characteristic times.

    Returns:
    np.array: The complex modulus for the generalized Maxwell model.

    """
    return Ge + np.sum(G[:, None] * tau[:, None] * 1.0j * omega / (1 + 1.0j * omega * tau[:, None]), axis=0)

######################MODELS FOR NLS FITTING START HERE##################################
def calculate_residual(model, G_hat):
    """
    Function to calculate the residual between a model output and the expected output, 
    handling both real and complex numbers and avoiding division by zero. 

    Parameters:
    model : numpy array
        Model output. Could be real or complex.
    G_hat : numpy array
        Expected output. Could be real or complex.

    Returns:
    residual : numpy array
        Residual between model and G_hat. 

    Notes:
    - NaN values in either model or G_hat are ignored in the calculation.
    - If G_hat is zero in a position, the residual in that position is set to zero.
    """

    # Remove NaNs from model and G_hat
    nan_mask = np.isnan(model) | np.isnan(G_hat)
    model = model[~nan_mask]
    G_hat = G_hat[~nan_mask]

    # Calculate residuals
    if np.iscomplexobj(G_hat):  # If G_hat is complex
        mask_real = G_hat.real != 0
        mask_imag = G_hat.imag != 0
        mask = mask_real | mask_imag
        residual_real = np.zeros_like(model.real)
        residual_imag = np.zeros_like(model.imag)
        residual_real[mask_real] = (model.real[mask_real] - G_hat.real[mask_real]) 
        residual_imag[mask_imag] = (model.imag[mask_imag] - G_hat.imag[mask_imag]) 
        residual = (residual_real + residual_imag) / 2.0
        #residual = (np.abs(model[mask])-np.abs(G_hat[mask]))
    else:  # If G_hat is real
        mask = G_hat != 0  # Create mask where G_hat is not zero
        residual = np.zeros_like(model)  # Initialize residual array
        residual[mask] = (model[mask] - G_hat[mask]) # Compute residual where G_hat is not zero

    return residual

def complex_modulus(params, omega, G_exp, arms, delta_t='', M=''):
    """
    Complex modulus for non-linear least square optimization

    Parameters
    ----------
    params : TYPE
        DESCRIPTION.
    omega : TYPE
        DESCRIPTION.
    G_exp : TYPE
        DESCRIPTION.
    arms : TYPE, optional
        DESCRIPTION. The default is 3.

    Returns
    -------
    residual : TYPE
        DESCRIPTION.

    """
    p = params.valuesdict()
    Ge = p['Ge']
    G = np.zeros(arms)
    tau = np.zeros(arms)
    for i in range(arms):
        G[i] = p[f'G{i+1}']
        tau[i] = p[f'tau{i+1}']

    model = Ge + np.sum(G[:, None] * tau[:, None] * 1j * omega / (1 + 1j * omega * tau[:, None]), axis=0)
    residual=calculate_residual(model,G_exp)    
    
    return residual


def g_hat(params, omega, G_hat, arms, Delta_t, M):
    """
    Model for non-linear optimization: Stress relaxation for a generalized maxwell model in the frequency domain
    For the context of finite and discrete time sequence. Thus the model needs M-seuquence length parameter and Delta_t
    """

    p = params.valuesdict()
    G_e = p['Ge']
    G_k = np.zeros(arms)
    tau_k = np.zeros(arms)
    for i in range(arms):
        G_k[i] = p[f'G{i+1}']
        tau_k[i] = p[f'tau{i+1}']
        
    term1 = G_e * (1 - np.exp(-1j * M * omega * Delta_t)) / (1 - np.exp(-1j * omega * Delta_t))
    term2 = np.sum([G_k[k] * (1 - np.exp(-M * Delta_t / tau_k[k]) * np.exp(-1j * M * omega * Delta_t)) / (1 - np.exp(-Delta_t / tau_k[k]) * np.exp(-1j * omega * Delta_t)) for k in range(arms)], axis=0)
    model = Delta_t * (term1 + term2) 
    residual = calculate_residual(model,G_hat)   
    
    return residual

def g_hat_ctft(params, omega, G_hat, arms, Delta_t='', M=''):
    """
    Model for non-linear optimization: Stress relaxation for a generalized maxwell model in the frequency domain
    For the context of continupus-time Fourier transform. Thus Delta_t and M-sequence length are ignored
    """

    p = params.valuesdict()
    G_e = p['Ge']
    G_k = np.zeros(arms)
    tau_k = np.zeros(arms)
    for i in range(arms):
        G_k[i] = p[f'G{i+1}']
        tau_k[i] = p[f'tau{i+1}']
         
    # Calculate the summation term
    model = G_e / (1j * omega)+ np.sum([G_k[i] * tau_k[i] / (1 + 1j * omega * tau_k[i]) for i in range(len(G_k))],axis=0)
    
    residual = calculate_residual(model,G_hat)   
    
    return residual

def func_g(params, t, G_exp, arms, delta_t='',M=''):
    """Model for NLS fitting: Stress relaxation for a generalized maxwell"""
    p = params.valuesdict()
    Ge = p['Ge']
    G = np.zeros(arms)
    tau = np.zeros(arms)
    for i in range(arms):
        G[i] = p[f'G{i+1}']
        tau[i] = p[f'tau{i+1}']
    
    model = Ge + np.sum(G[:, None] * np.exp (-t/tau[:, None]) , axis=0)
    
    residual = calculate_residual(model,G_exp)
    return  residual  # calculating the residual

######################MODELS FOR NLS FITTING END HERE##################################

######Next function is the wrap-around of the minimizer for NLS optimization##############
def nls_fit2(t, G_exp, arms,model, Ge, G_i, tau_i, delta_t, M,method_fit='leastsq'):
    """
    This function performs a non-linear fit based on any model given.
    """

    # create a set of Parameters
    params = Parameters()
    params.add('Ge', value=Ge, min=0)#,max=1.0e5)
    for i in range(arms):
        params.add(f'G{i+1}', value=G_i[i], min=G_i[i]/100)#, max=G_i[i]*1000)
        params.add(f'tau{i+1}', value=tau_i[i], min=tau_i[i]/100.0)#, max=tau_i[i]*1000.0)
    
    result = minimize(model, params, args=(t, G_exp, arms,delta_t,M), method=method_fit)

    Ge_c = result.params['Ge'].value
    G_c= np.zeros(arms)  
    tau_c = np.zeros(arms)
    for i in range(arms):
        G_c[i] = result.params[f'G{i+1}'].value
        tau_c[i]= result.params[f'tau{i+1}'].value
    
    return Ge_c, tau_c, G_c,result

def nls_fit(x, y, arms, model, Ge, G_i, tau_i, delta_t, M, 
            Ge_min=None, Ge_max=None, G_i_min=None, G_i_max=None, tau_i_min=None, tau_i_max=None, method_fit='least_squares'):
    """
    This function performs a non-linear fit on the given model using the provided parameters.

    Parameters
    ----------
    x : array
        Array representing the independent variable, typically either time or frequency

    y : array
        Array representing the experimentally obtained quantity, e.g. relaxation modulus

    arms : int
        Number of arms in the model.

    model : function
        Function representing the model to be fitted.

    Ge : float
        Initial guess for the equilibrium modulus.

    G_i : list of floats
        Initial guesses for the moduli of the different arms.

    tau_i : list of floats
        Initial guesses for the relaxation times of the different arms.

    delta_t : float
        Time step size.

    M : int
        Number of data points.

    Ge_min : float, optional
        Minimum possible value for the equilibrium modulus. Defaults to Ge_init/100.

    Ge_max : float, optional
        Maximum possible value for the equilibrium modulus. Defaults to Ge_init*100.

    G_i_min : list of floats, optional
        Minimum possible values for the moduli of the different arms. Defaults to respective G_i_init/100.

    G_i_max : list of floats, optional
        Maximum possible values for the moduli of the different arms. Defaults to respective G_i_init*100.

    tau_i_min : list of floats, optional
        Minimum possible values for the relaxation times of the different arms. Defaults to respective tau_i_init/100.

    tau_i_max : list of floats, optional
        Maximum possible values for the relaxation times of the different arms. Defaults to respective tau_i_init*100.

    method_fit : str, optional
        Method to be used for the fit. Defaults to 'leastsq' (Least-squares minimization).

    Returns
    -------
    Ge_c : float
        Fitted value for the equilibrium modulus.

    tau_c : array
        Fitted values for the relaxation times of the different arms.

    G_c : array
        Fitted values for the moduli of the different arms.
    result: object
        result of the fit with statistics about the fit
    """

    # set default values for min and max if they are not provided
    if Ge != 0:
        if Ge_min is None:
            Ge_min = Ge / 100
        if Ge_max is None:
            Ge_max = Ge * 100
    else:
        if Ge_min is None:
            Ge_min = 0.0
        if Ge_max is None:
            Ge_max = 1.0e3        
    if G_i_min is None:
        G_i_min = [gi/100 for gi in G_i]
    if G_i_max is None:
        G_i_max = [gi*100 for gi in G_i]
    if tau_i_min is None:
        tau_i_min = [ti/100 for ti in tau_i]
    if tau_i_max is None:
        tau_i_max = [ti*100 for ti in tau_i]

    # create a set of Parameters
    params = Parameters()
    params.add('Ge', value=Ge, min=Ge_min, max=Ge_max)
    for i in range(arms):
        params.add(f'G{i+1}', value=G_i[i], min=G_i_min[i], max=G_i_max[i])
        params.add(f'tau{i+1}', value=tau_i[i], min=tau_i_min[i], max=tau_i_max[i])
    
    result = minimize(model, params, args=(x,y, arms, delta_t, M), method=method_fit)

    Ge_c = result.params['Ge'].value
    G_c = np.zeros(arms)
    tau_c = np.zeros(arms)
    for i in range(arms):
        G_c[i] = result.params[f'G{i+1}'].value
        tau_c[i] = result.params[f'tau{i+1}'].value
    
    return Ge_c, tau_c, G_c, result





