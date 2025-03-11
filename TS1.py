# -*- coding: utf-8 -*-
"""
Created on Sat Mar 8 16:20:02 2025

@author: Magdalena Guida 
"""

import numpy as np
import matplotlib.pyplot as plt

fs =1000 # Frecuencia de muestreo
N = 1000   # Número de muestras
ts = 1 / fs  # Tiempo de muestreo
df = fs / N  # Resolución espectral    
# Definir funciónes seno
def func_sen(V_max=1, dc=0, frec=1, ph=0, nn=N, fs=fs):
    
    tt = np.arange(0, nn / fs, 1 / fs).flatten()
    
    xx = V_max * np.sin(2 * np.pi * frec * tt + ph).flatten() + dc
    
    return tt, xx

tt1, xx1 = func_sen(frec=1, fs=fs)
tt2, xx2 = func_sen(frec=10, fs=fs)
tt3, xx3 = func_sen(frec=2000, fs=fs)
# agregar cuenta de frecuencia 


# Caracteristicas del Grafico
plt.title("Señales Senoidales")
plt.plot(tt1, xx1, label="Frecuencia = 1 Hz", color = 'b')
plt.plot(tt2, xx2, label="Frecuencia = 8 Hz", color = 'r')
plt.plot(tt3, xx3, label="Frecuencia = 2000 Hz", color = 'y')
plt.legend(loc = 'lower left')
plt.xlabel("Tiempo [segundos]")
plt.ylabel("Amplitud [Volts]")
plt.grid(True)
plt.show()
    
