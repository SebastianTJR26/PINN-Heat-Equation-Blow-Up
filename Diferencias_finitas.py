import numpy as np
from Graficos_PINN import *


# Definir parámetros
L = 1  # Longitud del dominio (de 0 a 1) (se ajusto el el para el intervalo 0 a1)
Nx = 100  # Número de puntos espaciales  (mismos puntos para comparar con la solución exacta y la PINN)
Nt = 250  # Número de pasos temporales
dx = L / (Nx - 1)  # Tamaño de paso en x
dt = 5e-5  # Tamaño de paso en           deje dt se difinio asi, muy pequeño, porque si se hace más grande, hay inestabilidad númerica, o explota
x = np.linspace(0, 1, Nx)                #(Dominos de Dirichlet)
t = np.linspace(0, 1, Nt)
#x = np.linspace(-1, 1, Nx)  # Espacio
#t = np.linspace(0, dt*Nt, Nt)  # Tiempo

# Condición inicial
u0 = np.sin(np.pi*x)                     #(se cambia al nucleo de Dirichlet para comparar con la solución original)



# Resolver el caso (IV) u_t = u_xx 
def case_4():
    u = np.zeros((Nt, Nx))
    u[0, :] = u0  # Condición inicial

    for n in range(0, Nt-1):
        for i in range(1, Nx-1):
            u_xx = (u[n, i+1] - 2*u[n, i] + u[n, i-1]) / dx**2
            u[n+1, i] = u[n, i] + dt * u_xx #(u_xx + u[n, i]**(2))
                                  #0.001 explota
        # Condición de frontera
        u[n+1, 0] = u[n+1, 1] = 0

    return u

u_ψ= case_4()
# u_ψ es la solución dada por las diferencias finitas
mapa_de_calor_para_Diferencias_finitas_GRAF(u_ψ,Nt,dt)
np.savez("datos_dif_fin.npz", u_ψ=u_ψ, Nt=Nt, dt=dt)
print("datos ejecutados y guardados")





