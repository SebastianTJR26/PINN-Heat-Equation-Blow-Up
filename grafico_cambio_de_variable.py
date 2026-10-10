import numpy as np
import matplotlib.pyplot as plt

Izq = -10
Der = 150            #Esta si es toda la ventana

Arr = 1
Abj = 0

t0 = 0.3170

τ = np.linspace(Izq,Der,500)
f_τ1= t0*(1-np.exp(-τ))
f_τ2= t0*np.ones_like(τ)

plt.plot(τ,f_τ1,label='t(τ)=t0(1-e^(-τ))',color='green',lw=3)
plt.plot(τ,f_τ2,label='t0=0.3170',color='red',lw=3)
plt.xlabel('τ')
plt.ylabel('t(τ)')
plt.title('Convergencia del cambio de variable al t0 estimado de la explosión')
plt.legend()

#Esto es para decir que rango quieres que se mnuestre la gráfica, es decir si en "τ" quieres expandirlo más a la izquierda o derecha, o lo mismo en "f(τ)"
plt.xticks(np.arange(Izq, Der, 0.5))#Para modificar las escalas en x
plt.xlim(0,5) # Estoy delimitando lo que veo en la ventana, es decir solo quiero una zona para que se vea el prolongado hacia infinito 
plt.ylim(0,0.5) # Misma lógica
plt.grid(True) #Para tener cuadricula
plt.show()