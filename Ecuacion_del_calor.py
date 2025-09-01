import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import random

#Toda operación que lleve el tf, esta ejecutanode en GPU (porque ya CUDA se enlazo con mi grafica NVIDIA y da true)
#Toda operación con np, esta en CPU

 
from keras.saving import register_keras_serializable

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Para Python y NumPy
os.environ['PYTHONHASHSEED'] = '0' 
np.random.seed(0)

# Para TensorFlow (semilla global y determinismo en GPU)       #Estas son semillas, lo que buscan es que la EDP deje de varias y cuando se haga el mapa de calor, ya sea el mismo siempre, almenos que se modifiquen parametros como el alpha del oeprador no lineal
random.seed(0)
tf.random.set_seed(0)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

@register_keras_serializable()  # ¡Esta línea es crucial! 
#primer paso, se define como creo mi PINNN, cuantas capaz y neuronas
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class PINN(tf.keras.Model):  
    def __init__(self, layers):
        super(PINN, self).__init__()

        # Capas ocultas con activación 'tanh'
        self.hidden_layers = [tf.keras.layers.Dense(layer, activation='tanh') for layer in layers[:-1]] #original tanh

        # Capa de salida con activación None
        self.output_layer = tf.keras.layers.Dense(layers[-1], activation= 'relu') #None original relu

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



#segundo paso ,La función de pérdida es el núcleo de las PINNs. Combina tres componentes:
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#(1) Pérdida de la PDE: Mide cuánto se desvía la solución de la red neuronal de la ecuación diferencial. ∥∂u/∂t-α*∂²u/∂x²∥²
#...........................................................................................................................................................................................................................................................................
def EDP_loss(model, x, t, alpha,beta):
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

    return tf.reduce_mean((du_dt - alpha * d2u_dx2- beta*tf.pow(u,2))**2+100/(1+tf.norm(u, ord=2)))  # ||∂u/∂t - α * ∂²u/∂x²-  β *u²||²  

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



#tercer paso, generamos la data de la EDP (SEGUN LA ECUACION DEL CALOR) y definimos el entrenamiento
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#(1)Defino la data
#...........................................................................................................................................................................................................................................................................

#Parametros fisicos de la EDP
Nx=200
dt=0.001
Nt=200


 
# Coeficiente de difusión del calor (parametro físico)
alpha = tf.constant(1.0, dtype=tf.float32)
 
# Coeficiente de difusión del calor (parametro físico)                     #ESTO DE ACÁ, LOS ALPHA Y BETHA ESTAN SIENDO MUY IMPORTANTES, MI BETHA PEQUEÑO NO PERMITIA QUE DETECTARA EL  OP NO LINEAL, TODO SE ESTABA AJUSTANDO ERA A LA EDP SIN TENER ENCUENTA EL OP NO LINEAL
beta = tf.constant(1.0, dtype=tf.float32)                                 #POR ESO SIEMPRE SALIA LO MISMO, AUNQUE ESTAMOS AUN EN FASE DE PRUEBAS
 
 
# Discretización del espacio x en 100 puntos en el intervalo [0,1] ; el valor original era 100, ahora son 200 puntos de vizualisación
x = tf.convert_to_tensor(np.linspace(-1, 1, Nx), dtype=tf.float32)
                                                                             #aun necesito rastrearlos conn el type.watch, por alguna razon, no se rastrean bien
# Discretización del tiempo t en 100 puntos en el intervalo [0,1]
t = tf.convert_to_tensor(np.linspace(0, dt*Nt, Nt), dtype=tf.float32)
 
# Puntos iniciales en x (idénticos a x, porque representan la condición inicial en todo el dominio espacial)
#x_initial = tf.convert_to_tensor(np.linspace(0, 1, 100), dtype=tf.float32)
x_initial = x
 
# Puntos iniciales en t (todos ceros, porque representan t=0, el instante inicial)
t_initial = tf.zeros_like(x_initial, dtype=tf.float32)
#t_initial = t 
# Condición inicial u(x,0) = sin(pi * x), valores de u en t=0
#u_initial = tf.convert_to_tensor(np.sin(np.pi * np.linspace(0, 1, 100)), dtype=tf.float32)
u_initial = tf.convert_to_tensor(5*(1-np.linspace(-1, 1, Nx)**2),dtype=tf.float32) #LEONARDO
#u0 = np.where(np.abs(x) <= 1, 1.0, 0.0)
#u_initial = tf.convert_to_tensor(u0, dtype=tf.float32)
 
# Puntos de la frontera x=0 y x=1 (solo dos valores: 0 y 1, representan los extremos del dominio)
x_boundary = tf.convert_to_tensor(np.linspace(-1, 1, Nx), dtype=tf.float32)
                                                                                   #26/05/2025 Leonardo me pidio cambiar las dondiciones de frontera entre -1 a 1
# Valores de tiempo en la frontera (cubre todos los valores de t en [0,1])
t_boundary = tf.convert_to_tensor(np.linspace(0, 1, Nt), dtype=tf.float32)
 
# Condición de frontera: u(0,t) = u(1,t) = 0, esto significa que en los extremos el valor es siempre 0
u_boundary = 5*tf.ones_like(t_boundary, dtype=tf.float32)




#...........................................................................................................................................................................................................................................................................

#(2)defino mi modelo (cuantas capas y que optimizador quiero, usualmente es adam, y el número de epocas que se entrenara)
#model = PINN([20, 20, 20, 1])#la ultima capa siemrpe debe ser una neurona!! (1)
#model = PINN([50,25, 1]) #mejor explosión (2)
model = PINN([10, 15, 20, 25,30, 1])
#model = PINN([100,50,25, 1]) #sin el 25 da más temperatura, pero menos que el modelo anterior, y alborota más la fcn de perdida (3)
#model = PINN([50,1]) 
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4) #(4)
epochs = 3000
#...........................................................................................................................................................................................................................................................................


#(3)ENTRENAMIENTO! (ACTUALIZACIÓN, se adiciona en el código la forma de guardar los mejores pesos entrenados)
best_epoch = 0  #guardamos la epoca donde mejor fue entrenado el modelo
best_loss = 50000000.0 #guardamos el función de perdida más baja que dio, para saber cual fue el mejor entrenamiento, comenzamos en un valor muy alto para hacer comparación

w_pde = 0.6  # Prioriza la pérdida de la EDP
w_ic = 0.2
w_bc = 0.2

History_loss_1 = []
History_loss_2 = []
History_loss_3 = []
History_total_loss= []

for epoch in range(epochs):

    with tf.GradientTape() as tape:

        if epoch == 2400:
            optimizer.learning_rate.assign(5e-5)

        #elif epoch == 2000:
        #    optimizer.learning_rate.assign(0.00001)

        #elif epoch == 3000:
        #    optimizer.learning_rate.assign(0.000001)       


        loss_1 = EDP_loss(model, x, t, alpha,beta)

        loss_2 = Valores_Iniciales_loss(model, x_initial, t_initial, u_initial)

        loss_3 = Frontera_loss(model, x_boundary, t_boundary, u_boundary)

        History_loss_1.append(loss_1)

        History_loss_2.append(loss_2)

        History_loss_3.append(loss_3)

        PINN_loss = w_pde*loss_1 + w_ic*loss_2 + w_bc*loss_3

        History_total_loss.append(PINN_loss.numpy())  # Almacenar la pérdida total

        gradients = tape.gradient(PINN_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        loss = PINN_loss.numpy()
    
    if  loss < best_loss: #si encontro una función de perdida donde aprendio mejor (redujo más), guarde cuanto fue, en que época y guarde esos pesos que fueron los mejroes entrenados
        best_loss = loss
        best_epoch = epoch
        model.save_weights("mejores_pesos_pinn.weights.h5")        #(P)

    if epoch%100==0:
        xt = tf.concat([
        tf.reshape(x, (-1, 1)),
        tf.reshape(t, (-1, 1))
        ], axis=1)
        u_pred = model(xt)                                                  #al parecer, esto me esta diciendo en que época obtuve mi blow-up, tengo que ver si hubo una variación brusca
        intensidad= tf.reduce_max(tf.abs(u_pred)).numpy()
        print("intensidad en la epoca ", epoch," :",intensidad)

        

    
    print(f"Epoch {epoch}/{epochs},Loss_PINN: {loss:.5f}") 

print("holaa")
print("mejor epoca de entrenamiento:", best_epoch)
print("valor de la PINN_loss:", best_loss )

#---------------------------------------------------------------------------------------------
#PESOS
model.load_weights("mejores_pesos_pinn.weights.h5") #cargue los mejores pesos del entrenamiento  (P)
#---------------------------------------------------------------------------------------------


#cuarto paso, mostramos vizualisaciones de los datos
#---------------------------------------------------------------------------------------------

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
U_pred_np = U_pred.numpy()

# 5. Graficar la solución en 3D
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X_np, T_np, U_pred_np, cmap='hot')

# Etiquetas
ax.set_xlabel('Posición x')
ax.set_ylabel('Tiempo t')
ax.set_zlabel('Solución u(x,t)')
ax.set_title('Solución de la EDP con PINN, con U^3 Fuj')

plt.show()

#---------------------------------------------------------------------------------------------
#Crearemos un mapa de Calor para tener mejor vizualisación de la malla

plt.figure(figsize=(8, 6))
plt.contourf(X, T, U_pred, levels=100, cmap='inferno')  # Usa 'hot' para reflejar calor
plt.colorbar(label="Temperatura")
plt.xlabel("Posición x")
plt.ylabel("Tiempo t")
plt.title("Mapa de calor de la solución PINN, con U^2 op no lineal ARTICULO")
plt.show()



#---------------------------------------------------------------------------------------------
#ESTE ES UN MAPA DE CALOR DE CAPAZ POR CADA NEURONA, PARA VER SU ACTIVACIÓN
respuesta = input("ver activación de las neuronas? (S/n):")
if respuesta == 'S':
    for i, layer in enumerate(model.layers):
        weights = layer.get_weights()
        if weights:
            w = weights[0]  # solo los pesos (sin bias)
            plt.figure(figsize=(10, 2))
            plt.title(f'Pesos de la capa {i+1} - shape: {w.shape}')                          #(M)
            plt.imshow(w, aspect='auto', cmap='coolwarm')
            plt.colorbar()
            plt.show()



#---------------------------------------------------------------------------------------------
# Supongamos que ya tienes:
# T_vals: vector de tiempos de longitud M
# U_pred_np: array (M, N) con la solución PINN en la malla (t en filas, x en columnas)

# 1. Defino umbral pequeño, por ejemplo 1% de la amplitud inicial
epsilon = 0.01 * (U_pred_np.max() - U_pred_np.min())

# 2. Vector de máximos por cada tiempo
maximos = U_pred_np.max(axis=1)                                               #un ensayo de captura del tiempo en que explota la EDP

# 3. Tiempo inicial de calentamiento
idx = np.argmax(maximos > (maximos[0] + epsilon))
t_inicio = T_np[idx, 0]  # como T_np es malla, tomo la columna 0

print(f"Comienza a calentar en t ≈ {t_inicio:.4f}")


#---------------------------------------------------------------------------------------------
#GRAFICAS FUNCIONES DE PERDIDA
#Loss en EDP
plt.plot(History_loss_1,color='blue')
plt.title("Pérdida por época en la EDP")
plt.xlabel("Época")
plt.ylabel("Pérdida")
plt.show()

#Loss en valores iniciales
plt.plot(History_loss_2,color='orange')
plt.title("Pérdida por época en los valores iniciales")
plt.xlabel("Época")
plt.ylabel("Pérdida")
plt.show()

#Loss en frontera
plt.plot(History_loss_3,color='green')
plt.title("Pérdida por época en la frontera")
plt.xlabel("Época")
plt.ylabel("Pérdida")
plt.show()

#Loss TOTAL
plt.plot(History_total_loss,color='red')
plt.title("Pérdida por época TOTAL")
plt.xlabel("Época")
plt.ylabel("Pérdida")
plt.show()
#NOTAS

"""
versión 1

Esta es la primera versión de la EDP de calor que se realizo
* u_initial es SIN(πx) [0,1]
* Se genera una malla en 3D despues del entrenamiento "de colores azules"
* Esta es la PINN clasica, solo la EDP sin un operador no lineal y ninguna modificacion a la pinn_loss 
EDP loss original ||∂u/∂t - α * ∂²u/∂x²||²
"""


"""
versión 2

* u_initial ahora es e^-x [1,1], ahora converge el entrenamiento mejor
* La malla 3D ahora tiene colores rojos y amarillos, mostrando la intensidad de donde se ve más el calor , que es en las puntas de la montaña por asi decirlo
* se le agrego un opeador no lineal a la EDP, por tanto se una modificación en EDP Loss ||∂u/∂t - α * ∂²u/∂x²-  β *u²||²
* se realizo un mapa en 2D, para no ver la montaña, sino como un mapa de calor de la malla al estilo wilteck
"""


"""
versión 3

*se ajustaron comentarios que no se hicieron en "EDP original. op no lineal"
*se adiciono un mapa de calor para ver la activación de las neuronas por capa "OJO ESTO ES DESPUES DEL ENTRENAMIENTO"(M)
* en la proxima versión se adiconaran los comentarios de la (versión 1), más un comentario donde no estaba el operador no lineal
"""

"""
versión 4
*Se adicionaron los comentarios de la versión 1, más un mensaje de la EDP loss original
*Se aumento el entrenamiento de 500 a 1000
*Se adiciono el guardar los mejores pesos despues del entrenamiento (P)
*en el encabezado del código, se tuvo que agregar esta linea para poder guardar los pesos entrenados "register_keras_serializable()", más una libreria que no estaba en la versión 3
*neuronas muy activadas (color rojo intenso, valor positivo alto) neuronas negativas (valor negativo alto, ese valor no significa que la neurona este mal,sino que es un valor inverso que dio en el entrenamiento analogo a la neurona roja, son importantes tambien )
*neuronas claras(las que estan más cerca a cero, por ambos sentidos, + ó -, estas sin no sirven, no se estan activando)
*se guardaron tambien no solo los mejores pesos, sino en que época fue y el valor de la función de perdida que dio
"""



"""
versión 5

*Según el Teo de Fujita, la explosión de esta EDP se da para un  exponente p 1<p<=1+2/N, como esta es la ecuación de calor 1D, N=1, realizando la cuenta p<=3, el alpha del u se modifico de esta manera -> U^3, 
*se adicionaron semillas que quiere decir que los datos generados son fijos y no va a variar la EDP cuando se vea
*se copioo un código de captura del tiempo (no se si es la mejor opción y aun no lo entiendo bien, tal vez en un futuro se modifique o cambie por otro)
*Esto aun es una fase experimental, no estoy seguro si la explosión se esta haciendo de manera correcta, porque variando el p, sigue dando casi lo mismo, fuera del rango (1,3]
"""

"""
versión 6

*Experimentando y cambiando los valores de (α) y (β), estamos notando el blow up de la EDP, aun seguimos en fase de experimentación, el β quedo con un valor de 5.0 para que lo tuviera más encuenta la parte no lineal
*estamos variando el número de épocas si esto permite ver mejores resultados, por eso se veran diferentes çantidades de épocas en diferenets versiones del código
*tambien se esta experimentando con los dominios de (x) y (t), testo esta permitiendo ver cambios
*chatgpt arrojo un código de ver un cambio significativo en la explosión, aunque desconfio un poco de el.
*como los datos ya son fijos gracias a las semillas, ya tenemos confianza de que las avriaciones de calor es debido a los ajustes de lsom aprametros dichos anteriormente en esta versión, y no porque los datos varien ligeramente como se tenia antes.
"""


"""
versión 7

*Una vez hablado con Leonardo, me di cuenta que aun no tenia blow up ya que en el t=0 calentaba, pero de hay para adelante se enfriaba, lo cual no indicaba una buena señal
*Tocando las condiciones x in [-2,2] ; t in [-3,3] ; x_init in [-2,2] ; u_init = 2e^-x con su mismo intervalo estoy notando el blow up, lo raro es que veo calentamiento en la parte negativa, no se si explota tambien en ese dominio
*x_boundary esta entre [-2,2]

"""


"""
versión 8

* x in [0,30] ; t in [0,50]  ;x_init in [0,15] ;U^3; 

"""


"""
versión 9

*despues de muchas experimentaciones, tenemos devuelta a "x" y "t" en el dominio [0,1]
*La Ecuación ya ve un calentamiento en este dominio, se adiciono una variable "A" para la amplitud del operador no lineal de la exponencial
*Se adicionaron 3 pesos "w" a las funciones de perdida de EDP, Iniciales, Frontera, donde tiene más peso la función de perdida EDP
*se cambio la función de activación "tanh" por "relu"; 
*tanh satura en ±1, limitando la salida y frenando crecimientos fuertes. relu no satura para valores positivos, dejando que la red genere números grandes y capture mejor el blow-up.
*Tebemos ahora un ajuste de estimación en el dominio del blow-up (AA), tengo que estudiar más que hacen esas variables
*esperemos obtener un mejor blow-up despues
"""

"""
versión 10
*El Profesor Leonardo sugirio los siguientes cambios
*Si la condición inicial esta entre [-1,1], por qué no ajusto mis variables a ese entorno?
*En las condiciones de frontera, ya no estan registringidas a "0". Ahora estan por una condicion exponencial U(-1,t) y U(1,t)
*Alterando las épocas y el learning rate, los cambios del calor si aumentan más, la mejor es el modelo primero, aunque dejare el segundo
*Se quito la cota sugerida por ChatGPT
*Hasta ahora, la activación con "relu" es la que capta más datos
*Recuerda ahora variar los modelos y las épocas para obtener más calor
"""

"""
versión 11
*Se implementaron nuevos cambios sugeridos por el profesor Leonardo
*Las activaciones entre las capaz profundas son ahora tanh de nuevo, pero al salida es una relu, esto se hara para captar más información de la EDP
*Me indica que la frontera de t deberia ser entre [0,1], pero estoy teniendo mejores resultados en [-1,1], al igual que la amplitud A, me sugiere quitarla, pero la temperatura disminuye si hago eso
*La temperatura esta aumentando considerablemente en cada versión
*Queda pendiente dibujar las funciones de eprdida de loss_1, loss_2 y loss_3

"""
#el exponente esta en 2.5, no se a metido el cambio en el git

"""
versión 12
*Se a incluido la versión del exponente 2.5, este es el ejemplo que más se adecua a lo que indica el profesor Leonardo y por el momento ya no haremos más cambios en el aumento del blow-up

"""


"""
versión 13
*Se hicieron 3 arreglos para capturar las funciones de perdida (loss_1, loss_2, loss_3), para posteriormente plotearlas
*Hasta momento en mi PINN, la que más predomina es loss_1

"""

"""
versión 14
*Se encontro una mejor explosión con la red [50,50,1], llegando a 3 , tal vez el resultado de suavisar la función de perdida este en el número de neuronas y capas

"""

"""
versión 15
*Tenemos Ahora 4 diferentes arquitecturas de capaz y neuronas en 4 modelos, algunas dan más tempetatura, como (3), pero la fcn de perdida en edp sigue alborotada
*en (4) logramos desender la fcn de perdida, pero con una temperatura no tan fuerte, pero muy cerca de 3, aunque al quitarle la penultima capa aumenta T, pero se alborota de nuevo la fcn 

"""


"""
versión 16

*Se cambiaron los datos por todas las condiciones del articulo "On Blow-up Time and Rate Of The Numerical Solutions of The Semilinear Heat Equation with Reaction Term"
*Se añadieron unas condiciones al t_initial para que coincida con el u_0 , esto fue con ayuda del profesor Leonardo
*Los pesos estan priorizados del modo que tenga más encuenta la PDE (SI TENEMOS UNA TEMPERATURA MAYOR A LA QUE SE HABIA REGISTRADO)
*Esta pendiente modificar una forma de hacer más sencilla la arquitectura de la red, en donde pueda intercalar funciones de perdida entre las capaz, cambiar el learning rate en determinadas épcoas, y añadir el valor númerico  a la fcn residual->
*-> de perdia de la PDE loss_pde para ver como crece más, esto es sugerencia del profesor leonardo y debo ver el código online que tiene para ver como es
*Debo tratar de indpendicar la maya de puntos, ie que Nx sea diferente de Nt, porque python no me lo permite



"""


"""
versión 17
*Se adiciono a loss_pde n/(1+tf.norm(u, ord=2) con n>=10 para que crezca más, se nota que el cambio de temperatura aumenta
*El learning rate establecido al inico es de 5e-4
*En la época 2400 cambiamos el learning rate a 5e-5 porque ya se estaba estancando la fcn de perdida
*Con Nx=Nt=200 alcanza una temperatura de 5.7 hasta la época 3000, de hay para adelante,a si le coloque 6000 épocas se queda en 5.46, debe ser porque empieza a difundir la temperatura en la malla, es una hipotesis

"""