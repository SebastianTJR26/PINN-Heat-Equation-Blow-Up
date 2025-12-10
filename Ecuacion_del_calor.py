import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time
import os
import random
#Toda operación que lleve el tf, esta ejecutanode en GPU (porque ya CUDA se enlazo con mi grafica NVIDIA y da true)
#Toda operación con np, esta en CPU
#segundo paso, se define como creo mi PINNN, cuantas capaz y neuronas, y el 

os.environ['PYTHONHASHSEED'] = '0' 
np.random.seed(0)

# Para TensorFlow (semilla global y determinismo en GPU)       #Estas son semillas, lo que buscan es que la EDP deje de varias y cuando se haga el mapa de calor, ya sea el mismo siempre, almenos que se modifiquen parametros como el alpha del oeprador no lineal
random.seed(0)
tf.random.set_seed(0)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'


#primer paso, se define como creo mi PINNN, cuantas capaz y neuronas 
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class PINN(tf.keras.Model):  
    def __init__(self, layers):
        super(PINN, self).__init__()

        # Capas ocultas con activación 'tanh'
        self.hidden_layers = [tf.keras.layers.Dense(layer, activation='tanh') for layer in layers[:-1]]

        # Capa de salida con activación None
        self.output_layer = tf.keras.layers.Dense(layers[-1], activation=None)

    def call(self, inputs): # Forward propagation
        x = inputs[:, 0:1]  # Extrae la primera columna (posición x)
        t = inputs[:, 1:2]  # Extrae la segunda columna (tiempo t) 
        # Separa las entradas en x y t, ya que la EDP que queremos hallar está en función de u(x,t)
        # siendo "x" posición y "t" el tiempo
        # Son dos entradas porque esta EDP está en función de "x" y "t"

        CAPA_EDP = tf.concat([x, t], axis=1)  
        # Concatena "x" y "t". Estos son los DATOS de la EDP, 
        # se concatenan porque si no, tocaría hacer un entrenamiento separado para "x" y otro para "t"

        for layer in self.hidden_layers:  
            CAPA_EDP = layer(CAPA_EDP)  # Aplica cada capa densa

        CAPA_EDP = self.output_layer(CAPA_EDP)  
        # Aplica la capa de salida sin activación, ya que estamos modelando u(x,t)
            
        return CAPA_EDP  # Devuelve la salida de la red
  
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



#tercer paso ,La función de pérdida es el núcleo de las PINNs. Combina tres componentes:
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#(1) Pérdida de la PDE: Mide cuánto se desvía la solución de la red neuronal de la ecuación diferencial. ∥∂u/∂t-α*∂²u/∂x²∥²
#...........................................................................................................................................................................................................................................................................
def EDP_loss(model, x, t, alpha):
    # tf.GradientTape es una herramienta de TensorFlow para calcular derivadas (gradientes) de operaciones con respecto a sus variables.
    # with asegura que todas las operaciones dentro de él sean "grabadas" por GradientTape, para que luego se puedan calcular sus gradientes.
    # El parámetro persistent=True permite que la cinta (tape) sea reutilizable para calcular múltiples gradientes. 
    # Sin esto, la cinta se borraría después de calcular el primer gradiente.
    # tf.reduce_mean promedia estos errores para obtener una pérdida escalar.
    # tape.watch() le dice a TensorFlow que vigile o registre las operaciones que dependen de variables. 
    # Esto es necesario cuando variable no es un tensor entrenable (es decir, no es un parámetro de la red neuronal, 
    # como los pesos o biases), pero aún así necesitas calcular derivadas con respecto a esa variable.

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        tape.watch(t)

        # Combina los tensores x y t a lo largo del eje 1, en pocas palabras, crea la función u(x,t).
        # OJO: ESTA NO ES LA SOLUCIÓN ANALÍTICA O NUMÉRICA, es una (u) formada por la red.
        # Cambia u -> u_pred en caso de confusión.
        u = model(tf.concat([tf.reshape(x, (-1, 1)), tf.reshape(t, (-1, 1))], axis=1))

        # Diferenciación automática, calcula la derivada de la función u con respecto a t
        du_dt = tape.gradient(u, t)  

        # Diferenciación automática, calcula la derivada de la función u con respecto a x
        du_dx = tape.gradient(u, x)  

        # Diferenciación automática, calcula la segunda derivada de la función u con respecto a x
        d2u_dx2 = tape.gradient(du_dx, x)  

        # Las derivadas deben estar dentro del tape, porque TensorFlow está rastreando si se hace bien.
        # Si están fuera, podrían dar None.

    return tf.reduce_mean((du_dt - alpha * d2u_dx2)**2)  # ||∂u/∂t - α * ∂²u/∂x²||²

# Otra forma de hacerlo:
# u = model(tf.stack([x, t], axis=1))  # tf.stack([x, t], axis=1) y tape.watch([x, t])

#...........................................................................................................................................................................................................................................................................
 
#(2) Calculo con los valores inicales de la EDP para su función de perfida . ∥u(x,0)−u(x)∥² Mide cuánto se desvía la solución de la red neuronal de las condiciones iniciales.
#...........................................................................................................................................................................................................................................................................
def Valores_Iniciales_loss(model, x, t_initial, u_initial):
    #tape.watch([x, t_initial, u_initial])  # ¡Innecesario! ya que no se hara ninguna operación de diferencia finita, esto solo es cuando uno hace gradiente a cosas no declaradas con tensorflow, como numpy

    # Predecir u(x, t_initial) usando la red neuronal
    u = model(tf.stack([x, t_initial], axis=1))
    # Calcular la pérdida: (u - u_initial)²
    return tf.reduce_mean((u- u_initial)**2)

#...........................................................................................................................................................................................................................................................................


#(3) Perdida en la frontera: Mide cuánto se desvía la solución de la red neuronal de las condiciones de frontera. ∥u(0,t)-u_boundary(t)∥²+∥u(L,t)-u_boundary(t)∥²
#...........................................................................................................................................................................................................................................................................
def Frontera_loss(model,x_boundary,t_boundary,u_boundary):
    u_pred_0 = model(tf.stack([x_boundary[0] * tf.ones_like(t_boundary), t_boundary], axis=1))
    u_pred_L = model(tf.stack([x_boundary[1] * tf.ones_like(t_boundary), t_boundary], axis=1))
    
    loss_0 = tf.reduce_mean((u_pred_0 - u_boundary) ** 2)
    loss_L = tf.reduce_mean((u_pred_L - u_boundary) ** 2)
    return loss_0+loss_L
#...........................................................................................................................................................................................................................................................................


N_interior = 10000
#cuarto paso, creamos el entrenamiento
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#(1)Defino la data
#...........................................................................................................................................................................................................................................................................
# Discretización del espacio x en 100 puntos en el intervalo [0,1]
x = tf.convert_to_tensor(np.linspace(0, 1, 100), dtype=tf.float32)
                                                                             #aun necesito rastrearlos conn el type.watch, por alguna razon, no se rastrean bien
# Discretización del tiempo t en 100 puntos en el intervalo [0,1]
t = tf.convert_to_tensor(np.linspace(0, 1, 100), dtype=tf.float32)

# Puntos iniciales en x (idénticos a x, porque representan la condición inicial en todo el dominio espacial)
x_initial = tf.convert_to_tensor(np.linspace(0, 1, 100), dtype=tf.float32)

# Puntos iniciales en t (todos ceros, porque representan t=0, el instante inicial)
t_initial = tf.zeros_like(x_initial, dtype=tf.float32)

# Condición inicial u(x,0) = sin(pi * x), valores de u en t=0
u_initial = tf.convert_to_tensor(np.sin(np.pi * np.linspace(0, 1, 100)), dtype=tf.float32)

# Puntos de la frontera x=0 y x=1 (solo dos valores: 0 y 1, representan los extremos del dominio)
x_boundary = tf.convert_to_tensor([0, 1], dtype=tf.float32)

# Valores de tiempo en la frontera (cubre todos los valores de t en [0,1])
t_boundary = tf.convert_to_tensor(np.linspace(0, 1, 100), dtype=tf.float32)

# Condición de frontera: u(0,t) = u(1,t) = 0, esto significa que en los extremos el valor es siempre 0
u_boundary = tf.zeros_like(t_boundary, dtype=tf.float32)

# Coeficiente de difusión del calor (parametro físico)
alpha = tf.constant(0.01, dtype=tf.float32)
#...........................................................................................................................................................................................................................................................................

#(2)defino mi modelo (cuantas capas y que optimizador quiero, usualmente es adam, y el número de epocas que se entrenara)
model = PINN([20, 20, 20, 1])#la ultima capa siemrpe debe ser una neurona!!
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
epochs = 500
#...........................................................................................................................................................................................................................................................................

History_loss_1 = []
History_loss_2 = []
History_loss_3 = []
History_total_loss= []


#(3)ENTRENAMIENTO!
start = time.time() 
for epoch in range(epochs):
    with tf.GradientTape() as tape:
         loss_1 = EDP_loss(model, x, t, alpha)

         loss_2 = Valores_Iniciales_loss(model, x_initial, t_initial, u_initial)

         loss_3 = Frontera_loss(model, x_boundary, t_boundary, u_boundary)

         PINN_loss = loss_1 + loss_2 + loss_3

         History_loss_1.append(loss_1)
         History_loss_2.append(loss_2)
         History_loss_3.append(loss_3)
         History_total_loss.append(PINN_loss)


    gradients = tape.gradient(PINN_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    
    print(f"Epoch {epoch}/{epochs},Loss_PINN: {PINN_loss.numpy()}")
end = time.time()
print("Tiempo de ejecución:", end - start, "segundos")


#---------------------------------------------------------------------------------------------


# LA GRAFICA QUE MIRA EL TOTAL DE LAS FUNCIONES DE PERDIDA Y LAS ASINTOTICAS
def ultimos(x, n=100):
    n = min(len(x), n)
    start = max(0, len(x) - n)
    epocas = np.arange(start, len(x))
    return epocas, x[start:]

fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharex='col')
fig.suptitle("Funciones de Pérdida por Época", fontsize=16, y=0.98)

# --- EDP ---
axes[0,0].plot(History_loss_1, color='blue')
axes[0,0].set_title("EDP - Completo")
axes[0,0].set_ylabel("Pérdida")

ep_pde, y_pde = ultimos(History_loss_1, 100)
axes[0,1].plot(ep_pde, y_pde, color='blue')
axes[0,1].set_title("EDP - Últimas 100 épocas")

# --- Valores iniciales ---
axes[1,0].plot(History_loss_2, color='orange')
axes[1,0].set_title("Valores iniciales - Completo")
axes[1,0].set_ylabel("Pérdida")

ep_ic, y_ic = ultimos(History_loss_2, 100)
axes[1,1].plot(ep_ic, y_ic, color='orange')
axes[1,1].set_title("Valores iniciales - Últimas 100 épocas")

# --- Frontera ---
axes[2,0].plot(History_loss_3, color='green')
axes[2,0].set_title("Frontera - Completo")
axes[2,0].set_ylabel("Pérdida")

ep_bc, y_bc = ultimos(History_loss_3, 100)
axes[2,1].plot(ep_bc, y_bc, color='green')
axes[2,1].set_title("Frontera - Últimas 100 épocas")

# --- Total ---
axes[3,0].plot(History_total_loss, color='red')
axes[3,0].set_title("Total - Completo")
axes[3,0].set_xlabel("Época")
axes[3,0].set_ylabel("Pérdida")

ep_tot, y_tot = ultimos(History_total_loss, 100)
axes[3,1].plot(ep_tot, y_tot, color='red')
axes[3,1].set_title("Total - Últimas 100 épocas")
axes[3,1].set_xlabel("Época")

# Estética las variables que permiten plotear las superficies del mapa de calor y el 3D
for ax in axes.flat:
    ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# 1. Crear la malla con tf.meshgrid
X, T = tf.meshgrid(x, t)  # X representa las posiciones espaciales, T los tiempos

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
x_initial_np = x_initial.numpy()
t_initial_np = t_initial.numpy()
u_initial_np = u_initial.numpy()
t_boundary_np = t_boundary.numpy()
U_pred_np = U_pred.numpy()
x_np = X.numpy()
t_np = T.numpy()





fig = plt.figure(figsize=(14, 6))

# --- Subplot 1: Superficie 3D ---
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
surf = ax1.plot_surface(X_np, T_np, U_pred_np, cmap='jet')
ax1.set_xlabel('Posición x')
ax1.set_ylabel('Tiempo t')
ax1.set_zlabel('Solución u(x,t)')
ax1.set_title('Solución de la EDP con PINN')

# --- Subplot 2: Mapa de calor ---
ax2 = fig.add_subplot(1, 2, 2)
contour = ax2.contourf(X, T, U_pred, levels=100, cmap='jet')
fig.colorbar(contour, ax=ax2, label="Temperatura")
ax2.set_xlabel("Posición x")
ax2.set_ylabel("Tiempo t")
ax2.set_title("Mapa de calor de la solución PINN")

plt.tight_layout()
plt.show()


alpha_val = float(alpha.numpy()) if hasattr(alpha, "numpy") else float(alpha)

U_exact = np.sin(np.pi * X_np) * np.exp(- (np.pi**2) * alpha_val * T_np)

# (Opcional) Error relativo L2 en toda la malla
num = np.linalg.norm(U_pred_np - U_exact)
den = np.linalg.norm(U_exact) + 1e-12
print(f"Error relativo L2: {num/den:.3e}")


plt.figure()
plt.contourf(X_np, T_np, U_exact, 50)
plt.xlabel('x')
plt.ylabel('t')
plt.title('Solución exacta de la ecuación de calor')
plt.colorbar(label='u(x,t)')
plt.show()

# --- Un solo gráfico: mapa de calor del error absoluto ---
plt.figure(figsize=(8, 6))
plt.contourf(X_np, T_np, np.abs(U_pred_np - U_exact), levels=100)
plt.colorbar(label="|u_PINN - u_exacta|")
plt.xlabel("Posición x")
plt.ylabel("Tiempo t")
plt.title("Error absoluto entre PINN y solución analítica")
plt.tight_layout()
plt.show()


######## 

# ESTE ES EL GRÁFICO QUE PLOTEALA COLOCACIÓN DE LOS PUNTOS DEL DOMINIO DE LA EDP Y EN LA CONDICIÓN INICIAL


# Malla para puntos del PDE

plt.figure(figsize=(7, 5))

# Interior
plt.scatter(X, T, s=5, alpha=0.4, label="Interior")

# Condición inicial
plt.scatter(x_initial_np, t_initial_np, s=12, c='red', label="IC (t=0)")

# Fronteras x=-1 y x=1
plt.scatter( 0*np.ones_like(t_boundary_np), t_boundary_np, s=12, c='orange', label="BC x=-1")
plt.scatter( 1*np.ones_like(t_boundary_np), t_boundary_np, s=12, c='green', label="BC x=+1")

# Líneas verticales
plt.axvline( 0, color='orange', linestyle='--')
plt.axvline( 1, color='green', linestyle='--')

plt.xlabel("x")
plt.ylabel("t")
plt.title("Distribución de puntos en el dominio PDE")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

#Condición inicial curva más puntos

plt.figure(figsize=(7,5))
plt.plot(x_initial_np, u_initial_np, color="blue", label="u(x,0) = 5(1-x^2)")
plt.scatter(x_initial_np, u_initial_np, color="red", s=12, label="muestras IC")

plt.xlabel("x")
plt.ylabel("u(x,0)")
plt.title("Condición inicial: u(x,0) = 5(1-x^2)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()




fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ===============================
# SUBPLOT 1 — Distribución PDE
# ===============================

ax = axes[0]

# Interior
ax.scatter(X, T, s=5, alpha=0.4, label="Interior")

# Condición inicial
ax.scatter(x_initial_np, t_initial_np, s=12, c='red', label="IC (t=0)")

# Fronteras x=-1 y x=1
ax.scatter( 0*np.ones_like(t_boundary_np), t_boundary_np, s=12, c='orange', label="BC x=0")
ax.scatter( 1*np.ones_like(t_boundary_np), t_boundary_np, s=12, c='green', label="BC x=+1")

# Líneas verticales
ax.axvline(0, color='orange', linestyle='--')
ax.axvline(1, color='green', linestyle='--')

ax.set_xlabel("x")
ax.set_ylabel("t")
ax.set_title("Distribución de puntos en el dominio PDE")
ax.grid(True, alpha=0.3)
ax.legend()


# ===============================
# SUBPLOT 2 — Condición inicial
# ===============================

ax2 = axes[1]

ax2.plot(x_initial_np, u_initial_np, color="blue", label="u(x,0) = sin(pi*x)")
ax2.scatter(x_initial_np, u_initial_np, color="red", s=12, label="muestras IC")

ax2.set_xlabel("x")
ax2.set_ylabel("u(x,0)")
ax2.set_title("Condición inicial: u(x,0) = sin(pi*x)")
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.show()

#NOTAS
"""
Esta es la primera versión de la EDP de calor que se realizo
* u_initial es SIN(πx) [0,1]
* Se genera una malla en 3D despues del entrenamiento "de colores azules"
* Esta es la PINN clasica, solo la EDP sin un operador no lineal y ninguna modificacion a la pinn_loss
"""

"""
VERSIÓN ALTERNATIVA UNIFORME
*Ahora tiene la grafica de las 4 funciones de de perdida, las astintoticas, el mapa de calor y se incluyo un grafico 3D  en color jet en un solo plot
*Esta es la única versión que plote la colocación de los puntos del dominio y la condición inicial
*se incluyeron cosas posteriores como dejar la misma semilla para que el resultado de la EDP sea el mismo
*Tambien tiene la solución analítica
"""