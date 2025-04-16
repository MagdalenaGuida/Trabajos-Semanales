# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 18:06:50 2025

@author: magui
"""

# módulos y funciones a importar

import numpy as np
import matplotlib.pyplot as plt
#import scipy.signal as sig

def mi_funcion_sen (vmax, dc, ff, ph, N, fs):
    #fs frecuencia de muestreo (Hz)
    #N cantidad de muestras
    #ph: fase en radianes 

    ts = 1/fs # tiempo de muestreo o periodo
    tt=np.linspace (0, (N-1)*ts, N) #vector de tiempo con N ptos

    #generacion de la señal senoidal
    xx= dc + vmax*np.sin(2*np.pi*ff*tt + ph)
    #la señal debe generarse con la formula: x(t)=DC+Vmax*sen(2pift+fase)

    return tt, xx #devuelve vector tiempo y vector posicion

tt, xx = mi_funcion_sen(1.4, 0, 1, 0, 1000, 1000) # amp, dc, frec, N, fs
# Normalizacion
xn=xx/np.std(xx)  

#----------------------------------- Datos de la simulación---------------------------

fs =  1000 # frecuencia de muestreo (Hz)
N = 1000 # cantidad de muestras
# con 1000 para cada una normalizamos la resolucion espectral

ts = 1/fs # tiempo de muestreo
df = fs/N # resolución espectral

# Datos del ADC
B= 4 # bits (los elegimos entre todos)
Vf = 2 # rango simétrico de +/- Vf Volts



#--------------------------------------POTENCIA-----------------------------------
##1 de ganancia, fijarte el ancho de banda, y la potencia del radio 50 al cuadrado
# datos del ruido (potencia de la señal normalizada, es decir 1 W)
q = 2*Vf/(2**B)# paso de cuantización de q Volts = (Vmax - Vmin)/2^B
pot_ruido_cuant = q**2/12 # Watts
kn =1 # escala de la potencia de ruido analógico
pot_ruido_analog = pot_ruido_cuant * kn #potencia del ruido analogico?


    

# ------------------------------Señales -------------------------------------------

nn=np.random.normal(0,np.sqrt(pot_ruido_analog),N) #señal de ruido analogico
# media: 0, desvio: raiz de ruido analogico: ruido cuant * (K=1) = q/raiz(12)
analog_sig = xn # señal analógica sin ruido
sr = xn + nn # señal analógica de entrada al ADC (con ruido analógico)
srq = np.round(sr/q)*q
#Ej: si q=0.1 y sr=0,43, hago 0.43/0.1=4.3, redondeo para -> 4, entonces hago4*(q=0.1)=0.4
nq =  srq-sr # señal de ruido de cuantización || aca se hace la diferencia entre la señal y la señal cuantizada (ambas con ruido)

#------------------------Visualización de resultados-------------------------------

# --------------------------------Señal temporal---------------------------------

plt.figure()

plt.plot(tt, srq, lw=1, color='blue', label='$ s_Q = Q_{B,V_F}\{s_R\}$ (ADC out)')
plt.plot(tt, sr, color='g',alpha= 0.7, ls='dotted',marker='o',markerfacecolor='none',markeredgecolor='g',markersize=2, label='$ Sr= s + n $ (ADC in)')
plt.plot(tt, xn, color='yellow', ls='dotted', label='$s$ (analog)')

plt.title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q) )
plt.xlabel('tiempo [segundos]')
#plt.xlim(0, 0.02)
plt.ylabel('Amplitud [V]')
axes_hdl = plt.gca()
axes_hdl.legend()

# -----------------------------------------Espectro-------------------------------------


plt.figure()

#Uso la fft aplicada sobre señales en el tiempo para verlas en el espectro 
ft_SR = 1/N*np.fft.fft( sr)          #sr: analogica con ruido 
ft_Srq = 1/N*np.fft.fft( srq)        #srq: cuantizada
ft_As = 1/N*np.fft.fft( analog_sig)  #As: senoidal limpia
ft_Nq = 1/N*np.fft.fft( nq)          #Nq: ruido cuantizado= sr-srq
ft_Nn = 1/N*np.fft.fft( nn)          #Nn: ruido analogico 

# grilla de sampleo frecuencial
ff = np.linspace(0, (N-1)*df, N)   #vector de frecs desde 0 hasta fs-1 (en este caso pq N=fs) 

bfrec = ff <= fs/2    #bfrec: filtro booleano que te deja solo las frecs hasta fs/2 = frc de nyquist
Nnq_mean = np.mean(np.abs(ft_Nq)**2)
nNn_mean = np.mean(np.abs(ft_Nn)**2)

plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_As[bfrec])**2), color='yellow', ls='dotted', label='$ s $ (analog)' )
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_SR[bfrec])**2), ':g', label='$ s_R = s + n $ (ADC in)' )
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Srq[bfrec])**2), lw=2, label='$ s_Q = Q_{B,V_F}\{s_R\}$ (ADC out)' )

plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([nNn_mean, nNn_mean]) ), '--r', label= '$ \overline{n} = $' + '{:3.1f} dB (piso analogico)'.format(10* np.log10(2* nNn_mean)) )
plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([Nnq_mean, Nnq_mean]) ), '--c', label='$ \overline{n_Q} = $' + '{:3.1f} dB (piso digital)'.format(10* np.log10(2* Nnq_mean)) )
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Nn[bfrec])**2), ':r')
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Nq[bfrec])**2), ':c')



plt.title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q) )
plt.ylabel('Densidad de Potencia [dB]')
plt.xlabel('Frecuencia [Hz]')
axes_hdl = plt.gca()
axes_hdl.legend()


# --------------------------------------Histograma--------------------------------------

plt.figure(3)
bins = 10             #
plt.hist(nq.flatten()/(q), bins=bins)
plt.plot( np.array([-1/2, -1/2, 1/2, 1/2]), np.array([0, N/bins, N/bins, 0]), '--r' )
plt.title( 'Ruido de cuantización para {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q))
plt.xlabel('Pasos de cuantización (q) [V]')

