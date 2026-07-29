import os, random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.saving import register_keras_serializable
 
# =========================
# Reproducibilidad y precisión
# =========================
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(0)
random.seed(0)
tf.random.set_seed(0)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
 
# Usar float64 para estabilidad en segundas derivadas
tf.keras.backend.set_floatx('float64')
DTYPE    = tf.float64
P_EXP    = tf.constant(2,  dtype=DTYPE)
CLIP_MIN = tf.constant(-1e6, dtype=DTYPE)
CLIP_MAX = tf.constant( 1e6, dtype=DTYPE)
 
# =========================
# Dominio y parámetros físicos
# =========================
L_left, L_right = -1.0, 1.0
Nt = 300
dt = 1e-3
T_final = dt * Nt  # tiempo final coherente en todo el código
 
alpha = tf.constant(1.0, dtype=DTYPE)
beta  = tf.constant(1.0, dtype=DTYPE)
 
 
w_pde, w_ic, w_bc = 0.7, 0.2, 0.1  # CORRECCIÓN: Los pesos no deben ser cero
epochs     = 2100
batch_pde  = 1100
batch_ic   = 100
batch_bc   = 100
 
# =========================
# PINN
# =========================
@register_keras_serializable()
class PINN(tf.keras.Model):
    def __init__(self, layers_sizes):
        super().__init__()
        self.hidden = [
            tf.keras.layers.Dense(m, activation='softplus', dtype=DTYPE)
            for m in layers_sizes[:-1]
        ]
        # Capa de salida LINEAL (no forzar u>=0)
        self.out = tf.keras.layers.Dense(layers_sizes[-1], activation=None, dtype=DTYPE)
 
    def call(self, inputs):
        z = inputs
        for layer in self.hidden:
            z = layer(z)
        return self.out(z)  # u(x,t)
 
# =========================
# Condición inicial y muestreo
# =========================
def u0_fun(x):
    # u(x,0) = 5(1 - x^2)
    return 5.0 * (1.0 - tf.square(x))
 
def sample_pde_points(batch_pde):
    # Uniforme en (x,t) ∈ [-1,1] × [0, T_final]
    x = tf.random.uniform((batch_pde, 1), minval=L_left, maxval=L_right, dtype=DTYPE)
    t = tf.random.uniform((batch_pde, 1), minval=0.0,    maxval=T_final,  dtype=DTYPE)
    return x, t
 
def sample_ic_points(batch_ic):
    # t=0, x ~ U[-1,1]
    x  = tf.random.uniform((batch_ic, 1), minval=L_left, maxval=L_right, dtype=DTYPE)
    t0 = tf.zeros_like(x, dtype=DTYPE)
    u0 = u0_fun(x)
    return x, t0, u0
 
def sample_bc_points(batch_bc):
    # x = ±1, t ~ U[0,T_final], u = 0 en la frontera (coherente con la IC)
    t  = tf.random.uniform((batch_bc, 1), minval=0.0, maxval=T_final, dtype=DTYPE)
    xL = tf.ones_like(t, dtype=DTYPE) * L_left
    xR = tf.ones_like(t, dtype=DTYPE) * L_right
    u_b = tf.zeros_like(t, dtype=DTYPE)  # CORRECCIÓN: u debe ser 0 en la frontera
    return xL, xR, t, u_b
 
# =========================
# Pérdidas (derivadas fuera del contexto)
# =========================
def pde_loss(model, x, t, alpha, beta):
    # r = u_t - alpha u_xx - beta u^{2.5}
    # tape2 "envuelve" a tape1 para poder derivar u_x respecto a x (obtener u_xx)
    with tf.GradientTape() as tape2:
        tape2.watch([x, t])
        with tf.GradientTape() as tape1:
            tape1.watch([x, t])
            u = model(tf.concat([x, t], axis=1))  # u(x,t)
 
        # Pedimos ambos gradientes en UNA llamada (tape1 no persistente)
        u_x, u_t = tape1.gradient(u, [x, t])
 
    # Segunda derivada espacial
    u_xx = tape2.gradient(u_x, x)
 
    # Asegurar dtypes consistentes con el de u
    alpha_ = tf.cast(alpha, u.dtype)
    beta_  = tf.cast(beta,  u.dtype)
    p_exp  = tf.cast(P_EXP, u.dtype)
 
    u_clip = tf.clip_by_value(
        u,
        tf.cast(CLIP_MIN, u.dtype),
        tf.cast(CLIP_MAX, u.dtype)
    )
    nonlin = tf.pow(u_clip, p_exp)
 
    resid = u_t - alpha_ * u_xx - beta_ * nonlin
    return tf.reduce_mean(tf.square(resid))
 
def ic_loss(model, x0, t0, u0):
    u_pred0 = model(tf.concat([x0, t0], axis=1))
    return tf.reduce_mean(tf.square(u_pred0 - u0))
 
def bc_loss(model, xL, xR, t_b, u_b):
    # CORRECCIÓN: Recibir xL y xR como argumentos en lugar de crearlos dentro
    uL = model(tf.concat([xL, t_b], axis=1))
    uR = model(tf.concat([xR, t_b], axis=1))
    return tf.reduce_mean(tf.square(uL - u_b)) + tf.reduce_mean(tf.square(uR - u_b))
 
# =========================
# Modelo, optimizador y pesos de pérdidas
# =========================
model = PINN([30, 30, 30, 30, 1])
 
lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=[int(0.33*epochs), int(0.66*epochs)],
    values=[tf.constant(1e-2, DTYPE), tf.constant(1e-3, DTYPE), tf.constant(5e-4, DTYPE)]
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_sched)
 
 
 
# =========================
# Muestreo inicial y figura solicitada
# =========================
x_pde, t_pde           = sample_pde_points(batch_pde)
x_ic,  t_ic, u_ic      = sample_ic_points(batch_ic)
xL, xR, t_bc, u_bc     = sample_bc_points(batch_bc)
 
# Curva suave de la IC para trazar
x_line = tf.linspace(tf.constant(L_left, DTYPE), tf.constant(L_right, DTYPE), 400)
u0_line = u0_fun(tf.reshape(x_line, (-1,1)))
 
# Figura: distribución (x,t) + condición inicial
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
 
# Distribución (x,t)
axs[0].scatter(x_pde.numpy(), t_pde.numpy(), s=6,  alpha=0.35, label='PDE collocation', color='C0')
axs[0].scatter(x_ic.numpy(),  t_ic.numpy(),  s=12, alpha=0.9,  label='IC (t=0)', color='C3')
axs[0].scatter(xL.numpy(),    t_bc.numpy(),  s=12, alpha=0.8,  label='BC x=-1', color='C2', marker='|')
axs[0].scatter(xR.numpy(),    t_bc.numpy(),  s=12, alpha=0.8,  label='BC x=+1', color='C4', marker='|')
axs[0].set_xlabel('x'); axs[0].set_ylabel('t')
axs[0].set_title('Distribución de puntos (PDE, IC y BC)')
axs[0].set_xlim(L_left-0.05, L_right+0.05)
axs[0].set_ylim(-0.02*T_final, 1.02*T_final)
axs[0].grid(True, alpha=0.25)
axs[0].legend(loc='upper right', fontsize=9)
 
# Condición inicial u(x,0)
axs[1].plot(x_line.numpy(), u0_line.numpy(), lw=2, label=r'$u(x,0)=5(1-x^2)$')
axs[1].scatter(x_ic.numpy(), u_ic.numpy(), s=12, alpha=0.8, color='C3', label='muestras IC')
axs[1].set_xlabel('x'); axs[1].set_ylabel('u(x,0)')
axs[1].set_title('Condición inicial (curva y muestras)')
axs[1].grid(True, alpha=0.25)
axs[1].legend(loc='best', fontsize=9)
 
plt.tight_layout(); plt.show()
 
# =========================
# Entrenamiento
# =========================
best_epoch = -1
best_loss  = np.inf
 
Hist_pde, Hist_ic, Hist_bc, Hist_total = [], [], [], []
 
for epoch in range(1, epochs+1):
    # Remuestreo en cada época (recomendado)
    x_pde, t_pde     = sample_pde_points(batch_pde)
    x_ic, t_ic, u_ic = sample_ic_points(batch_ic)
    xL, xR, t_bc, u_bc = sample_bc_points(batch_bc)  # CORRECCIÓN: Obtener todos los valores
 
    with tf.GradientTape() as tape:
        L_pde = pde_loss(model, x_pde, t_pde, alpha, beta)
        L_ic  = ic_loss(model, x_ic,  t_ic,  u_ic)
        L_bc  = bc_loss(model, xL, xR, t_bc, u_bc)  # CORRECCIÓN: Pasar xL y xR
        total = w_pde*L_pde + w_ic*L_ic + w_bc*L_bc
 
    grads = tape.gradient(total, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
 
    # Historial
    Hist_pde.append(float(L_pde.numpy()))
    Hist_ic.append(float(L_ic.numpy()))
    Hist_bc.append(float(L_bc.numpy()))
    Hist_total.append(float(total.numpy()))
 
    # Mejor modelo
    if total.numpy() < best_loss:
        best_loss = float(total.numpy())
        best_epoch = epoch
        model.save_weights("mejores_pesos_pinn.weights.h5")
 
    # Logging ocasional
    if epoch % 200 == 0:
        print(f"Epoch {epoch}/{epochs} | L_pde={L_pde.numpy():.3e}  L_ic={L_ic.numpy():.3e}  "
              f"L_bc={L_bc.numpy():.3e}  Total={total.numpy():.3e}")
 
print("Entrenamiento finalizado.")
print("Mejor época:", best_epoch)
print("Mejor pérdida total:", best_loss)
model.load_weights("mejores_pesos_pinn.weights.h5")
 
# =========================
# (Opcional) Curvas de pérdida
# =========================
plt.figure(figsize=(10,5))
plt.plot(Hist_pde,   label='PDE')
plt.plot(Hist_ic,    label='IC')
plt.plot(Hist_bc,    label='BC')
plt.plot(Hist_total, label='Total', linewidth=2)
plt.yscale('log'); plt.grid(True, alpha=0.3)
plt.xlabel('Época'); plt.ylabel('Pérdida'); plt.legend(); plt.tight_layout(); plt.show()
 
# =========================
# Gráficos de pérdidas detalladas (CORRECCIÓN: Usar las listas correctas)
# =========================
fig, axs = plt.subplots(4, 2, figsize=(12, 12))
 
# Primer plot: Pérdida en la EDP (todas las épocas)
axs[0, 0].plot(Hist_pde, color='blue')
axs[0, 0].set_title("Pérdida por época en la EDP")
axs[0, 0].set_xlabel("Época")
axs[0, 0].set_ylabel("Pérdida")
axs[0, 0].grid(True, alpha=0.3)
 
# Segundo plot: Pérdida en la EDP (últimas 1200 épocas)
axs[0, 1].plot(Hist_pde[-1200:], color='blue')
axs[0, 1].set_title("Pérdida por época en la EDP (Últimas 1200 épocas)")
axs[0, 1].set_xlabel("Época")
axs[0, 1].set_ylabel("Pérdida")
axs[0, 1].grid(True, alpha=0.3)
 
# Tercer plot: Pérdida en valores iniciales (todas las épocas)
axs[1, 0].plot(Hist_ic, color='orange')
axs[1, 0].set_title("Pérdida por época en los valores iniciales")
axs[1, 0].set_xlabel("Época")
axs[1, 0].set_ylabel("Pérdida")
axs[1, 0].grid(True, alpha=0.3)
 
# Cuarto plot: Pérdida en valores iniciales (últimas 1200 épocas)
axs[1, 1].plot(Hist_ic[-int(epochs*0.8):], color='orange')
axs[1, 1].set_title("Pérdida por época en los valores iniciales (Últimas 1200 épocas)")
axs[1, 1].set_xlabel("Época")
axs[1, 1].set_ylabel("Pérdida")
axs[1, 1].grid(True, alpha=0.3)
 
# Quinto plot: Pérdida en la frontera (todas las épocas)
axs[2, 0].plot(Hist_bc, color='green')
axs[2, 0].set_title("Pérdida por época en la frontera")
axs[2, 0].set_xlabel("Época")
axs[2, 0].set_ylabel("Pérdida")
axs[2, 0].grid(True, alpha=0.3)
 
# Sexto plot: Pérdida en la frontera (últimas 1200 épocas)
axs[2, 1].plot(Hist_bc[-int(epochs*0.8):], color='green')
axs[2, 1].set_title("Pérdida por época en la frontera (Últimas 1200 épocas)")
axs[2, 1].set_xlabel("Época")
axs[2, 1].set_ylabel("Pérdida")
axs[2, 1].grid(True, alpha=0.3)
 
# Séptimo plot: Pérdida total de la PINN (todas las épocas)
axs[3, 0].plot(Hist_total, color='red')
axs[3, 0].set_title("Pérdida total de la PINN")
axs[3, 0].set_xlabel("Época")
axs[3, 0].set_ylabel("Pérdida")
axs[3, 0].grid(True, alpha=0.3)
 
# Octavo plot: Pérdida total de la PINN (últimas 1200 épocas)
axs[3, 1].plot(Hist_total[-int(epochs*0.8):], color='red')
axs[3, 1].set_title("Pérdida total de la PINN (Últimas 1200 épocas)")
axs[3, 1].set_xlabel("Época")
axs[3, 1].set_ylabel("Pérdida")
axs[3, 1].grid(True, alpha=0.3)
 
# Ajustar espaciado entre subplots
plt.tight_layout()
 
# Mostrar la figura
plt.show()
 
 
 
# =========================
# Cuarto paso: Visualizaciones de los datos (Mapas de calor)
# =========================
 
# Crear una malla más fina para visualización
x_vis = tf.linspace(L_left, L_right, 300)
t_vis = tf.linspace(0.0, T_final, 300)
 
# 1. Crear la malla con tf.meshgrid
X, T = tf.meshgrid(x_vis, t_vis)  # X representa las posiciones espaciales, T los tiempos
 
# 2. Aplanar para formar los inputs al modelo
X_flat = tf.reshape(X, [-1])
T_flat = tf.reshape(T, [-1])
input_tensor = tf.stack([X_flat, T_flat], axis=1)  # (N, 2) donde N = 10000
 
# 3. Evaluar la PINN
U_pred = model(input_tensor)  # u(x,t) predicho por la red
U_pred = tf.reshape(U_pred, X.shape)  # Volver a la forma (100,100)
 
# 4. Convertir a NumPy para graficar
X_np = X.numpy()
T_np = T.numpy()
U_pred_np = U_pred.numpy()
 
fig = plt.figure(figsize=(12, 6))
 
# Primer subplot: Gráfico 3D
ax1 = fig.add_subplot(121, projection='3d')
surf = ax1.plot_surface(X_np, T_np, U_pred_np, cmap='jet', antialiased=True)
ax1.set_xlabel('Posición x')
ax1.set_ylabel('Tiempo t')
ax1.set_zlabel('Solución u(x,t)')
ax1.set_title('Solución 3D de la EDP con PINN')
fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=20)
 
# Segundo subplot: Mapa de calor 2D
ax2 = fig.add_subplot(122)
contour = ax2.contourf(X_np, T_np, U_pred_np, levels=100, cmap='jet')
ax2.set_xlabel("Posición x")
ax2.set_ylabel("Tiempo t")
ax2.set_title("Mapa de calor de la solución PINN")
fig.colorbar(contour, ax=ax2, label="Temperatura")
 
# Ajustar espaciado y mostrar
plt.tight_layout()
plt.show()