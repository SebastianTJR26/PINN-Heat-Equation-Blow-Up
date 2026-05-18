import numpy as np
from Graficos_PINN import *


# Definir parámetros
L = 2  # Longitud del dominio (de -1 a 1)
Nx = 100  # Número de puntos espaciales
Nt = 250  # Número de pasos temporales
dx = 2*L / (Nx - 1)  # Tamaño de paso en x
dt = 5e-4  # Tamaño de paso en t
x = np.linspace(-1, 1, Nx)  # Espacio
t = np.linspace(0, dt*Nt, Nt)  # Tiempo

# Condición inicial
u0 = 5 * (1 - x**2)


# Resolver el caso (IV) u_t = u_xx + u^2
def case_4():
    u = np.zeros((Nt, Nx))
    u[0, :] = u0  # Condición inicial

    for n in range(0, Nt-1):
        for i in range(1, Nx-1):
            u_xx = (u[n, i+1] - 2*u[n, i] + u[n, i-1]) / dx**2
            u[n+1, i] = u[n, i] + dt * (u_xx + u[n, i]**(2))

        # Condición de frontera
        u[n+1, 0] = u[n+1, -1] = 0

    return u

u_ψ= case_4()
# u_ψ es la solución dada por las fierencias finitas
mapa_de_calor_para_Diferencias_finitas_GRAF(u_ψ,Nt,dt)
np.savez("datos_dif_fin.npz", u_ψ=u_ψ, Nt=Nt, dt=dt)
print("datos ejecutados y guardados")





