# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 17:56:04 2025

@author: magui
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Definir la función de transferencia H(s) = V0 / Vi
W0 = 1
Q=3 #Defino con valores arbitrarios, la unica condicion es Q>1/2 para tener polos complejos conjugados
numerador = [W0/Q, 0]  # Numerador 
denominador = [1, W0/Q, W0]  # Denominador 

# función transferencia
sistema = signal.TransferFunction(numerador, denominador)


f = np.logspace(-1, 2, 400)  #  (de 0.1 a 100 rad/s)
w = 2 * np.pi * f

# Respuesta al módulo y a la fase
w, mag, fase_grados = signal.bode(sistema, w)

# Convertir fase de grados a radianes
fase_radianes = np.deg2rad(fase_grados)

# Grafico rta de módulo 
plt.figure(figsize=(7, 4))
plt.ylim(-50,5)
plt.semilogx(w, mag, "r")  #  escala logarítmica
plt.title("Respuesta de Módulo")
plt.xlabel("Frecuencia [rad/s]")
plt.ylabel("Magnitud [dB]")
plt.axhline(y=-3, color="grey", linestyle="--")
plt.axvline(x=0.85, color="grey", linestyle="--") # W1
plt.axvline(x=1.18, color="grey", linestyle="--") # W2, donde W2 - W1 = ancho de banda
plt.grid(True)
plt.figtext(0.5, -0.07, 'Grafico I: Respuesta de modulo, filtro pasivo pasa banda RLC segundo orden', ha='center', fontsize=8, fontstyle='italic')
plt.show()

# Grafico rta de fase 
plt.figure(figsize=(3, 3))
plt.ylim(-2 , 2)  
plt.semilogx(w, fase_radianes, "r")
plt.title("Respuesta de Fase")
plt.xlabel("Frecuencia [rad/s]")
plt.ylabel("Fase [rad]")
plt.axhline(y=1.6, color="grey", linestyle="--")
plt.axhline(y=-1.6, color="grey", linestyle="--")
plt.grid(True)
plt.figtext(0.5, -0.09, 'Grafico II: Respuesta de fase, filtro pasivo pasa banda RLC segundo orden', ha='center', fontsize=8, fontstyle='italic')
plt.show()
plt.show()


# Definir la función de transferencia H(s) = V0 / Vi
W0 = 1
Q=0.5
 #Defino con valores arbitrarios 
numerador = [(-1),0,0]  # Numerador 
denominador = [(-1), W0/Q, W0**2]  # Denominador 

# función transferencia
sistema = signal.TransferFunction(numerador, denominador)


f = np.logspace(-1, 2, 300)  #  (de 0.1 a 100 rad/s)
w = f* 2* np.pi

# Respuesta al módulo y a la fase
w, mag, fase_grados = signal.bode(sistema, w)

# Convertir fase de grados a radianes
fase_radianes = np.deg2rad(fase_grados)

#Grafico rta de módulo 
plt.figure(figsize=(7, 4))
plt.semilogx(w, mag, "r")  #  escala logarítmica
plt.title("Respuesta de Módulo")
plt.xlabel("Frecuencia [rad/s]")
plt.ylabel("Magnitud [dB]")
plt.axhline(y=-3, color="grey", linestyle="--")
plt.axvline(x=2.5, color="grey", linestyle="--") # Wc
plt.grid(True)
plt.figtext(0.5, -0.07, 'Grafico III: Respuesta de modulo, filtro pasivo pasa alto RLC segundo orden', ha='center', fontsize=8, fontstyle='italic')
plt.show()
plt.show()