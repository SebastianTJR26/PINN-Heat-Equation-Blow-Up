import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def distribucion_dominio_y_condicion_inicial_GRAF(x,τ,minvalx,maxvalx,minvalτ,maxvalτ,x_in,τ_in,u0_func,x_line,u0_line,x_bnL,x_bnR,τ_bn,T_blow_up):

	# Figura: distribución (x,τ) + condición inicial
	fig, axs = plt.subplots(1, 2, figsize=(12, 5))
	# Distribución (x,t)
	axs[0].scatter(x.numpy(), τ.numpy(), s=6, alpha=0.8,  label='PDE collocation', color='C0')
	axs[0].scatter(x_in.numpy(),  τ_in.numpy(),  s=6, alpha=0.8,  label='IC (t=0)', color='C3')
	axs[0].scatter(x_bnL.numpy(),    τ_bn.numpy(),  s=6, alpha=0.8,  label='BC x=-1', color='C2', marker='|')
	axs[0].scatter(x_bnR.numpy(),    τ_bn.numpy(),  s=6, alpha=0.8,  label='BC x=+1', color='C2', marker='|')
	axs[0].set_xlabel('x'); axs[0].set_ylabel('t')
	axs[0].set_title('Distribución de puntos (PDE, IC y BC)')
	axs[0].set_xlim(minvalx-0.05, maxvalx+0.05)
	axs[0].set_ylim(-0.02*minvalτ, 1.02*maxvalτ)#(-0.02*T_final, 1.02*T_final)
	axs[0].grid(True, alpha=0.25)
	axs[0].legend(loc='upper right', fontsize=9)
	
	# Condición inicial u(x,0)
	axs[1].plot(x_line.numpy(), u0_line.numpy(), lw=2, label=r'$u(x,0)=5(1-x^2)$')
	axs[1].scatter(x_in.numpy(), u0_func.numpy(), color="red", s=12, label="muestras IC")
	axs[1].set_xlabel("x")
	axs[1].set_ylabel("u(x,0)")
	axs[1].set_title("Condición inicial: u(x,0) = 5(1-x^2)")
	axs[1].grid(True, alpha=0.3)

	axs[1].legend()
	

	plt.tight_layout(); plt.show()

def ultimos(x, asintota):
	n = min(len(x), asintota)
	start = max(0, len(x) - asintota)
	epocas = np.arange(start, len(x))
	return epocas, x[start:]

def funciones_de_perdida_GRAF(x,asintota,History_loss_1,History_loss_2,History_loss_3,History_total_loss):
	fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharex='col')
	fig.suptitle("Funciones de Pérdida por Época", fontsize=16, y=0.98)

	# --- EDP ---
	axes[0,0].plot(History_loss_1, color='blue')
	axes[0,0].set_title("EDP - Completo")
	axes[0,0].set_ylabel("Pérdida")

	ep_pde, y_pde = ultimos(History_loss_1, asintota)
	axes[0,1].plot(ep_pde, y_pde, color='blue')
	axes[0,1].set_title("EDP - Últimas "+ str(asintota) +" épocas")

	# --- Valores iniciales ---
	axes[1,0].plot(History_loss_2, color='orange')
	axes[1,0].set_title("Valores iniciales - Completo")
	axes[1,0].set_ylabel("Pérdida")

	ep_ic, y_ic = ultimos(History_loss_2, asintota)
	axes[1,1].plot(ep_ic, y_ic, color='orange')
	axes[1,1].set_title("Valores Iniciales - Últimas "+ str(asintota) +" épocas")

	# --- Frontera ---
	axes[2,0].plot(History_loss_3, color='green')
	axes[2,0].set_title("Frontera - Completo")
	axes[2,0].set_ylabel("Pérdida")

	ep_bc, y_bc = ultimos(History_loss_3, asintota)
	axes[2,1].plot(ep_bc, y_bc, color='green')
	axes[2,1].set_title("Frontera - Últimas "+ str(asintota) +" épocas")

	# --- Total ---
	axes[3,0].plot(History_total_loss, color='red')
	axes[3,0].set_title("Total - Completo")
	axes[3,0].set_xlabel("Época")
	axes[3,0].set_ylabel("Pérdida")

	ep_tot, y_tot = ultimos(History_total_loss, asintota)
	axes[3,1].plot(ep_tot, y_tot, color='red')
	axes[3,1].set_title("Total - Últimas "+ str(asintota) +" épocas")
	axes[3,1].set_xlabel("Época")

	# Estética
	for ax in axes.flat:
		ax.grid(True, alpha=0.3)

	plt.tight_layout(rect=[0, 0, 1, 0.96])
	plt.show()




def mapa_de_calor_y_grafico_3D_para_la_PINN_GRAF(u_φ,X_np,T_np):
	fig = plt.figure(figsize=(14, 6))

	# --- Subplot 1: Superficie 3D ---
	ax1 = fig.add_subplot(1, 2, 1, projection='3d')
	surf = ax1.plot_surface(X_np, T_np, u_φ, cmap='jet')
	ax1.set_xlabel('Posición x')
	ax1.set_ylabel('Tiempo t')
	ax1.set_zlabel('Solución u(x,t)')
	ax1.set_title('Solución de la EDP con PINN + DF, con U^2' )

	# --- Subplot 2: Mapa de calor ---
	ax2 = fig.add_subplot(1, 2, 2)
	contour = ax2.contourf(X_np, T_np, u_φ, levels=100, cmap='jet')
	fig.colorbar(contour, ax=ax2, label="Temperatura")
	ax2.set_xlabel("Posición x")
	ax2.set_ylabel("Tiempo t")
	ax2.set_title("Mapa de calor de la solución PINN + DF, con U^2")

	plt.tight_layout()
	plt.show()




def mapa_de_calor_para_Diferencias_finitas_GRAF(u_ψ,Nt,dt):
	plt.subplot(1, 1, 1)
	plt.title("Mapa de calor solución diferencias finitas: $u_t = u_{xx} - u^{2}$")
	plt.imshow(u_ψ,origin='lower', aspect='auto', cmap='jet', extent=[-1, 1, 0, Nt*dt])
	plt.colorbar(label='u(x,t)')
	plt.xlabel('x')
	plt.ylabel('t')

	plt.tight_layout()
	plt.show()



def PINN_vs_Diferencias_finitas_GRAF(u_φ,X_np,T_np):
	data_diff = np.load("datos_dif_fin.npz")
	u_ψ = data_diff["u_ψ"]
	Nt = data_diff["Nt"]
	dt = data_diff["dt"]
	mapa_de_calor_para_Diferencias_finitas_GRAF(u_ψ,Nt,dt)


	fig = plt.figure(figsize=(15, 5))

	# Primer subplot: Solución PINN
	ax1 = fig.add_subplot(131)
	contour1 = ax1.contourf(X_np, T_np, u_φ, levels=100, cmap='jet')
	ax1.set_xlabel("Posición x")
	ax1.set_ylabel("Tiempo t")
	ax1.set_title("Mapa de calor de la solución PINN + DF")
	fig.colorbar(contour1, ax=ax1, label="Temperatura")

	# Segundo subplot: Diferencias finitas
	ax2 = fig.add_subplot(132)
	contour2 = ax2.contourf(X_np, T_np, u_ψ, levels=100, cmap='jet')  # Asumiendo que u_case_2 tiene misma dimensión
	ax2.set_xlabel("Posición x")
	ax2.set_ylabel("Tiempo t")
	ax2.set_title("Mapa de calor solución diferencias finitas")
	fig.colorbar(contour2, ax=ax2, label="Temperatura")

	# Tercer subplot: Error absoluto
	ax3 = fig.add_subplot(133)  # ¡CORREGIDO: era 132, debe ser 133!
	contour3 = ax3.contourf(X_np, T_np, np.abs(u_φ - u_ψ), levels=100, cmap='jet')  # CORREGIDO: u_case_2 - U_pred_np
	ax3.set_xlabel("Posición x")
	ax3.set_ylabel("Tiempo t")
	ax3.set_title("Error absoluto punto a punto")
	fig.colorbar(contour3, ax=ax3, label="Temperatura")

	# Ajustar espaciado y mostrar
	plt.tight_layout()
	plt.show()