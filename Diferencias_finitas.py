import numpy as np
from Graficos_PINN import *


# Definir parámetros
maxvalτ= 1
t0=0.3170
#L = 2  # Longitud del dominio (de -1 a 1)
Nx = 100 # Número de puntos espaciales
Nτ = 325*maxvalτ  # Número de pasos temporales la cantidad de puntos tiene que ir creciendo con el máximo del intervalo

x = np.linspace(-1, 1, Nx)  # Espacio
τ = np.linspace(0, maxvalτ, Nτ)  # Tiempo
"""
Δx = 2*L / (Nx - 1)  # Tamaño de paso en x      normal
Δτ = 0.002 # Tamaño de paso en t
"""
Δx = 2*np.abs(x[0]-x[1])  # Tamaño de paso en x, por alguna razón si no se mutiplica por 2 se inestabiliza
#Δτ = np.abs(τ[0]-τ[1])
Δτ = 0.002 # Tamaño de paso en τ, deberia ser la diferencia ya que es el paso exacto en el que avanza, pero por alguna razón tambien se inestabiliza, y con 0.002 no


print("\n")
print("delta τ: ",Δτ)
print("\n")
print("delta x: ", Δx)
print("\n")

# Condición inicial
u0 = 5 * (1 - x**2)


# Resolver el caso (IV) u_t = u_xx + u^2
def case_4():
    u = np.zeros((Nτ, Nx))
    u[0, :] = u0  # Condición inicial

    for n in range(0, Nτ-1):
        for i in range(1, Nx-1):
            u_xx = (u[n, i+1] - 2*u[n, i] + u[n, i-1]) / Δx**2
            factor = t0*np.exp(-τ[n])
            u[n+1, i] = u[n, i] + Δτ *factor*(u_xx + u[n, i]**(2))

        # Condición de frontera
        u[n+1, 0] = u[n+1, -1] = 0

    return u

u_ψ= case_4()
# u_ψ es la solución dada por las fierencias finitas
mapa_de_calor_para_Diferencias_finitas_GRAF(u_ψ,maxvalτ)
np.savez("datos_dif_fin.npz", u_ψ=u_ψ, maxvalτ=maxvalτ)
print("datos ejecutados y guardados")





