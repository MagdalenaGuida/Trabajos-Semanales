# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 12:06:49 2025

@author: magui
"""

  # -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 21:25:39 2025

@author: magui
"""

import numpy as np 
import matplotlib.pyplot as plt 
import scipy.signal as sig 


#%%DEFINO PARAMETROS
SNR =3 #[dB]
R =200  #cant de realizaciones 
N = 1000 # cant de muestras ----------> esto me da dim NxR 
N_2 = 10*N
fs= 1000
ts = 1/fs
a1 = np.sqrt(2) #amplitud normalizda
# w0 = np.pi/2 
w0 = fs/4
#df = 2*np.pi/N
df= fs/N
df2 = fs/N_2
pot_ruido = 10**(-SNR/10)
verificacion = -10 * np.log10(pot_ruido)
# print(verificacion)
DS_potruido = np.sqrt(pot_ruido)       #OBS OBS 


vect_t = np.arange(0, 1, 1/N).reshape(N, 1)
tt = np.tile(vect_t, (1, R))


fr = np.random.uniform(-1/2,1/2,size = (1,R))

vect_w1 = w0 + fr*df

#SEÑAL LIMPIA 
S = a1*np.sin(vect_w1 * tt * 2 * np.pi) # mult matricia entre w1 y tt = S[1000,200]=[N,R]


#%%  Definicion X // X ventaneados

#DEF RUIDO 
vect_nn=np.random.normal(0, DS_potruido, size=(N,R))  #señal de ruido analogico
# media: 0, desvio: raiz de ruido analogico
# print(vect_nn.shape)
nn = vect_nn         # matriz de ruido 

#SEÑAL CON RUIDO  VENTANA RECT
X = S + nn    
# VENTANA FLATTOP
X_flattop = X * sig.windows.flattop(N).reshape(-1,1)
# VENATANA BLACKHARRIS
X_blackharris = X * sig.windows.blackmanharris(N).reshape(-1,1)
# VENTANA HAMMING
X_hamming = X * sig.windows.hamming(N).reshape(-1,1)


#%%  Def FFTs   (+ GRAFICO) 

X_fft = np.fft.fft(X, axis = 0) / N
X_flattop_fft = np.fft.fft(X_flattop, axis=0)/N
X_blackharris_fft = np.fft.fft(X_blackharris, axis=0)/N
X_hamming_fft = np.fft.fft(X_hamming, axis=0)/N


# grilla de sampleo frecuencial
ff = np.linspace(0, (N-1)*df, N)   #vector de frecs desde 0 hasta fs-1 (en este caso pq N=fs) 

bfrec = ff <= fs/2    #bfrec: filtro booleano que te deja solo las frecs hasta fs/2 = frc de nyquist

plt.figure()
plt.plot(ff[bfrec], 10* np.log10(2* np.abs(X_blackharris_fft[bfrec])**2 ) )

plt.title('Espectro de Potencia de la Señal- 3dB ')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Potencia (dB)')
plt.grid(True)  # Habilitamos las líneas de grilla
axes_hdl = plt.gca()

plt.show() 
    


#%% DEFINIR ESTIMADOR DE AMPLITUD = a1

#valor verdadero = a1
a1_rect = np.abs(X_fft[N//4, :])
a1_flattop = np.abs( X_flattop_fft[N//4, :])
a1_blackharris =  np.abs(X_blackharris_fft[N//4, :])
a1_hamming = np.abs(X_hamming_fft[N//4, :])

# Calculo el valor promedio de cada estimador 
a1_rect_esp = np.mean(a1_rect)
a1_flattop_esp = np.mean(a1_flattop)
a1_blackharris_esp = np.mean(a1_blackharris)
a1_hamming_esp = np.mean(a1_hamming)


# SESGO 
 
# Sesgo:  Calculo la dif entre el valor maximo de la señal y el valor real 

sesgo_rect_a1 = a1_rect_esp - a1 
sesgo_flattop_a1 = a1_flattop_esp-a1
sesgo_blackharris_a1 = a1_blackharris_esp-a1
sesgo_hamming_a1 = a1_hamming_esp - a1 


#VARIANZA 

var_a1_rect = np.var(a1_rect)
var_a1_flattop = np.var(a1_flattop)
var_a1_blackharris = np.var(a1_blackharris)
var_a1_hamming = np.var(a1_hamming)

#%% Definir estimador de Frecs = omega

X_fft_abs = np.abs(X_fft[:N//2, :])              # np.abs = valor absoluto de cada valor desde 0 a N/2, filas y todas las columnas, hago solo N72 pq al ser espejado la info se repite 
flattop_fft_abs = np.abs(X_flattop_fft[:N//2, :]) 
blackharris_fft_abs = np.abs(X_blackharris_fft[:N//2, :])
hamming_fft_abs = np.abs(X_hamming_fft[:N//2, :])

omega_rect = np.argmax(X_fft_abs, axis=0) * df  # argmax, axis=0 te tira la posicion de maximo por columna, y dsp se lo mult por la df para N/2
omega_flattop = np.argmax(flattop_fft_abs, axis =0)* df
omega_bh = np.argmax(blackharris_fft_abs, axis =0)* df 
omega_hamming = np.argmax(hamming_fft_abs, axis=0)* df 

#SESGO 

#valor real = w0
omega_rect_esp = np.mean(omega_rect)   #mean = valor promedio 
omega_bh_esp = np.mean(omega_bh)
omega_flattop_esp = np.mean(omega_flattop)
omega_hamming_esp = np.mean(omega_hamming)

sesgo_rect_om = omega_rect_esp- w0
sesgo_bh_om = omega_bh_esp -w0
sesgo_flattop_om = omega_flattop_esp -w0
sesgo_hamming_om = omega_hamming_esp -w0

#VARIANZA

var_rect_om = np.var(omega_rect)
var_flattop_om = np.var(omega_flattop)
var_bh_om = np.var(omega_bh)
var_hamming_om = np.var(omega_hamming)

#%% GRAFICOS 

# Histograma omega
plt.figure()
plt.hist(omega_rect, bins=10, color='red', alpha=0.5, label="Estimador sin ventanear")
plt.hist(omega_bh, bins=10, color='green',alpha=0.5, label="Estimador ventana Blackmanharris")
plt.hist(omega_hamming, bins=10, color='blue',alpha=0.5, label="Estimador ventana blackmanharris")
plt.hist(omega_flattop, bins=10, color='pink', alpha=0.5, label="Estimador ventana flattop")
plt.title("Histograma de frecuencias estimadas - SNR 3 dB")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Cantidad de ocurrencias")
plt.grid(True)
plt.legend()
 
plt.show()

# Histograma a1
plt.figure()
plt.hist(a1_rect, bins=10, color='red', alpha=0.5, label="Estimador sin ventanear") #Bins: resolucion espectral del histograma; conteo relativo. ANCHURA de los valores.
plt.hist(a1_blackharris, bins=10, color='green', alpha=0.5, label="Estimador ventana Blackmanharris")
plt.hist(a1_hamming, bins=10, color='blue', alpha=0.5, label="Estimador ventana Hamming")
plt.hist(a1_flattop, bins=10, color='pink', alpha=0.5, label="Estimador ventana flattop")
plt.title("Histograma de amplitudes estimadas - SNR 3 dB")
plt.xlabel("Amp")
plt.ylabel("Cantidad de ocurrencias")
plt.grid(True)
plt.legend()

plt.show()


# Crear tabla con los encabezados fijos
tabla = [
    ["Ventana ",     "SESGO a1",      "VAR a1",      "SESGO omega1",      "VAR omega1"],
    ["Rectangular", sesgo_rect_a1, var_a1_rect, sesgo_rect_om, var_rect_om],
    ["Flattop", sesgo_flattop_a1,  var_a1_flattop, sesgo_flattop_om, var_flattop_om],
    ["Blackmanharris", sesgo_blackharris_a1,  var_a1_blackharris, sesgo_bh_om, var_bh_om],
    ["Hamming", sesgo_hamming_a1,  var_a1_hamming, sesgo_hamming_om, var_hamming_om]
]

# Mostrar como imagen
fig, ax = plt.subplots(figsize=(12, 2 + len(tabla)*0.5))
ax.axis('tight')
ax.axis('off')
plt.title("Sesgo y Varianza para a1 y omega1 - 3dB", fontsize=11, pad=1)
table = ax.table(cellText=tabla, loc='center', cellLoc='center')

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.6)

plt.tight_layout()
plt.show()



