import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# 1. Resolver con diferencias finitas (método numérico)
# ==========================================================
N = 50  # puntos internos
x = np.linspace(0, 1, N+2)  # incluye 0 y 1
h = x[1] - x[0]

# Matriz del sistema Au = b
A = np.zeros((N, N))
b = np.zeros(N)

for i in range(N):
    A[i, i] = -2
    if i > 0:
        A[i, i-1] = 1
    if i < N-1:
        A[i, i+1] = 1
    b[i] = -(np.pi**2) * np.sin(np.pi * x[i+1]) * h**2

# Resolver el sistema lineal
u_inner = np.linalg.solve(A, b)
u_num = np.zeros(N+2)
u_num[1:-1] = u_inner  # condiciones de frontera en 0

# ==========================================================
# 2. Definir la PINN
# ==========================================================
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(1,)),
    tf.keras.layers.Dense(20, activation='tanh'),
    tf.keras.layers.Dense(20, activation='tanh'),
    tf.keras.layers.Dense(1)
])

# Función de pérdida (datos numéricos + física)
def pinn_loss(x_tf, u_num_tf):
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(x_tf)
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(x_tf)
            u_pred = model(x_tf)
        u_x = tape1.gradient(u_pred, x_tf)
    u_xx = tape2.gradient(u_x, x_tf)

    # Residuo de la EDP
    f = -(np.pi**2) * tf.sin(np.pi * x_tf)
    res = u_xx - f

    # Condiciones de frontera
    bc0 = model(tf.constant([[0.0]]))
    bc1 = model(tf.constant([[1.0]]))

    # Error con solución numérica (método de diferencias finitas)
    data_loss = tf.reduce_mean((u_pred - u_num_tf)**2)

    # Error físico
    physics_loss = tf.reduce_mean(res**2)

    # Error de frontera
    bc_loss = bc0**2 + bc1**2

    return data_loss + physics_loss + bc_loss

# Optimizador
optimizer = tf.keras.optimizers.Adam(0.01)

# Datos numéricos como tensores (usamos los mismos puntos de la malla)
x_tf = tf.convert_to_tensor(x.reshape(-1,1), dtype=tf.float32)
u_num_tf = tf.convert_to_tensor(u_num.reshape(-1,1), dtype=tf.float32)

# Entrenamiento
for epoch in range(2000):
    with tf.GradientTape() as tape:
        loss_value = pinn_loss(x_tf, u_num_tf)
    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss = {float(loss_value.numpy()):.6f}")

# Predicción final
u_pred = model(x_tf).numpy()

plt.plot(x, u_num, label="Solución numérica (DF)")
plt.plot(x, u_pred, '--', label="PINN")
plt.legend()
plt.show()
