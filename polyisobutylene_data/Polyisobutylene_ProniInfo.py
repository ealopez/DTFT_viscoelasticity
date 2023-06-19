# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 16:56:38 2017

@author: Enrique Alejandro
"""

import numpy as np
import matplotlib.pyplot as plt
from quique import*

a = np.loadtxt('Polyisobutylene_PronyData.txt', delimiter=',', skiprows=1)

b = np.loadtxt('RelModulus_Polyisobutylene.txt', delimiter=',',skiprows=1)

t_th = b[:,0]
G_th = b[:,1]

tau = a[:,0]

G = a[:,1]

figure(1)
plt.plot(tau,G)
plt.xscale('log')
plt.yscale('log')

t = log_tw(1.0e-12,1.0e6,10)

G_t = np.zeros(size(t))

for i in range(size(t)):
    G_t[i] = sum(G[:]*exp(-t[i]/tau[:]))

figure(2)
plt.plot(t, G_t, 'b')
plt.plot(t_th, G_th, 'g*') 
plt.xscale('log')
plt.yscale('log')

np.savetxt('PIB.txt', array((tau, G)).T, delimiter='\t')