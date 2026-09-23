import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Graficos_PINN import *
import numpy as np
import time
import os
import math
import random

#Toda operación que lleve el tf, esta ejecutanode en GPU (porque ya CUDA se enlazo con mi grafica NVIDIA y da true)
#Toda operación con np, esta en CPU

 
from keras.saving import register_keras_serializable
condicion = input("Activar muestreo adaptativo (S/n):")
if condicion=='S':
    print("\n\nResiduo adaptativo")
    print("\n")
    puntero_condicion_adaptattiva = ['S']
else:
    print("\n\nResiduo no adaptativo")
    print("\n")
    puntero_condicion_adaptattiva = ['n']   

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Para Python y NumPy
os.environ['PYTHONHASHSEED'] = '0' 
np.random.seed(0)

# Para TensorFlow (semilla global y determinismo en GPU)       #Estas son semillas, lo que buscan es que la EDP deje de varias y cuando se haga el mapa de calor, ya sea el mismo siempre, almenos que se modifiquen parametros como el alpha del oeprador no lineal
random.seed(0)
tf.random.set_seed(0)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
tf.keras.backend.set_floatx('float64')

@register_keras_serializable()  # ¡Esta línea es crucial! 
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
        # Separa las entradas en x y t, ya que la EDP que queremos hallar está en función de u_φ(x,t)
        # siendo "x" posición y "t" el tiempo
        # Son dos entradas porque esta EDP está en función de "x" y "t"

        CAPA_EDP = tf.concat([x, t], axis=1)  
        # Concatena "x" y "t". Estos son los DATOS de la EDP, 
        # se concatenan porque si no, tocaría hacer un entrenamiento separado para "x" y otro para "t"

        for layer in self.hidden_layers:  
            CAPA_EDP = layer(CAPA_EDP)  # Aplica cada capa densa

        CAPA_EDP = self.output_layer(CAPA_EDP)  
        # Aplica la capa de salida sin activación, ya que estamos modelando u_φ(x,t)
            
        return CAPA_EDP  # Devuelve la salida de la red
  
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#segundo paso ,La función de pérdida es el núcleo de las PINNs. Combina tres componentes:
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#(1) Pérdida de la PDE: Mide cuánto se desvía la solución de la red neuronal de la ecuación diferencial. ∥∂u/∂t-α*∂²u/∂x²∥²
#...........................................................................................................................................................................................................................................................................
#antes, las condiciones iniciales y la frontera estaban en stack, no concatenadas, ahora ya si estan concatenadas

#La solución de la Ecuación Diferencial dada por la PINN se denoratara por: ---(u_φ)---

def EDP_loss(model, x, t,w_ti):
    # tf.GradientTape es una herramienta de TensorFlow para calcular derivadas (gradientes) de operaciones con respecto a sus variables.
    # with asegura que todas las operaciones dentro de él sean "grabadas" por GradientTape, para que luego se puedan calcular sus gradientes.
    # El parámetro persistent=True permite que la cinta (tape) sea reutilizable para calcular múltiples gradientes. 
    # Sin esto, la cinta se borraría después de calcular el primer gradiente.
    # tf.reduce_mean promedia estos errores para obtener una pérdida escalar.
    # tape.watch() le dice a TensorFlow que vigile o registre las operaciones que dependen de variables. 
    # Esto es necesario cuando variable no es un tensor entrenable (es decir, no es un parámetro de la red neuronal, 
    # como los pesos o biases), pero aún así necesitas calcular derivadas con respecto a esa variable.

 
     #como esto no se esta derivando, no es necesario su verificación
    #print(w_ti)

    
    with tf.GradientTape(persistent=True) as tape:

        tape.watch(x)
        tape.watch(t)

        # Combina los tensores x y t a lo largo del eje 1, en pocas palabras, crea la función u(x,t).
        # OJO: ESTA NO ES LA SOLUCIÓN ANALÍTICA O NUMÉRICA, es una (u) formada por la red.
        # Cambia u_φ -> u_pred en caso de confusión.
        u_φ = model(tf.concat([tf.reshape(x, (-1, 1)), tf.reshape(t, (-1, 1))], axis=1))

        # Diferenciación automática, calcula la derivada de la función u con respecto a t
        du_dt = tape.gradient(u_φ, t)  

        # Diferenciación automática, calcula la derivada de la función u con respecto a x
        du_dx = tape.gradient(u_φ, x)  

        # Diferenciación automática, calcula la segunda derivada de la función u con respecto a x
        d2u_dx2 = tape.gradient(du_dx, x)  

        # Las derivadas deben estar dentro del tape, porque TensorFlow está rastreando si se hace bien.
        # Si están fuera, podrían dar None.

    residuo_edp = du_dt-d2u_dx2-tf.pow(u_φ,2)   
    

    if puntero_condicion_adaptattiva[0]=='S':
        # Sumar y elevar al cuadrado
        Sum = tf.reduce_sum(tf.pow(w_ti,2))
        # Normalizar:
        Norm_wti = w_ti/tf.math.sqrt(Sum)
        # Multiplicar: Norm_wti * tf.pow(residuo_edp,2)
        pesos = Norm_wti * tf.pow(residuo_edp,2)
        # el residuo
        RES = tf.reduce_sum(pesos)/ tf.reduce_sum(Norm_wti)
        #RETURN 2 CON PONDERACIÓN ADAPTATIVA
        
        #L = tf.reduce_mean(w_ti) * tf.reduce_mean((residuo_edp)**2)
        #L = w_ti * tf.pow(residuo_edp,2)
        #RES =tf.reduce_mean(L)/ tf.reduce_mean(w_ti)
        #RES = L/ tf.reduce_mean(w_ti)
        #RES = tf.reduce_mean((residuo_edp)**2)
        #print(RES)
     
        return RES #-> (Σ w_ti * ||R||²)/ Σw_ti ESTA BIEN,  SI CUMPLE BIEN LA DEFINICIÓN
        #return tf.reduce_sum(w_ti * tf.square(residuo_edp)) / tf.reduce_sum(w_ti) #-> (Σ w_ti * √R)/ Σw_ti ESTA MAL, NO ESTA SACANDO LA NORMA AL CUADRADO
    
    else:
        RES = tf.reduce_mean((residuo_edp)**2)
        #print(RES)
        return RES #||∂u/∂t -  ∂²u/∂x²-  *u²||²
    

# tf.reduce_mean((residuo_edp)**2) = tf.square(residuo_edp)

# Otra forma de hacerlo:
# u = model(tf.stack([x, t], axis=1))  # tf.stack([x, t], axis=1) y tape.watch([x, t])

#...........................................................................................................................................................................................................................................................................
 
#(2) Cálculo con los valores inicales de la EDP para su función de perfida . ∥u(x,0)−u(x)∥² Mide cuanto se desvía la solución de la red neuronal de las condiciones iniciales.
#...........................................................................................................................................................................................................................................................................
def Valores_Iniciales_loss(model, x_in, t_in, u0_func):
    #tape.watch([x, t_initial, u_initial])  # ¡Innecesario! ya que no se hara ninguna operación de diferencia finita, esto solo es cuando uno hace gradiente a cosas no declaradas con tensorflow, como numpy

    # Predecir u(x, t_initial) usando la red neuronal
    u_φ0 = model(tf.concat([x_in, t_in], axis=1))
    # Calcular la pérdida: (u - u_initial)²
    return (tf.reduce_mean((u_φ0- u0_func)**2))

#...........................................................................................................................................................................................................................................................................


#(3) Perdida en la frontera: Mide cuánto se desvía la solución de la red neuronal de las condiciones de frontera. ∥u(0,t)-u_boundary(t)∥²+∥u(L,t)-u_boundary(t)∥²
#...........................................................................................................................................................................................................................................................................
def Frontera_loss(model, x_bnL, x_bnR , t_bn, u_bn_func):
    u_φL = model(tf.concat([x_bnL, t_bn], axis=1))
    u_φR = model(tf.concat([x_bnR, t_bn], axis=1))
    
    loss_L = tf.reduce_mean((u_φL - u_bn_func) ** 2)
    loss_R = tf.reduce_mean((u_φR - u_bn_func) ** 2)
    return loss_L+loss_R
    #TANTO LA FUNCION DE PERDIDA DE LA FRONTERA COMO LOS VALORES INICIALES ESTABAN EN tf.stack Y NO tf.contac, TAL VEZ ESO INFLUYERA EN EL ERROR DEL TENSOR DIFERENTE
#...........................................................................................................................................................................................................................................................................

def Interfaz_espacio_loss(P1model,P2model,x,t,x_mid):
    intercepcion_x = tf.ones_like(x, dtype=tf.float64) * x_mid
    PINN1 = P1model(tf.concat([intercepcion_x, t], axis=1))  #loss_interface(intercepcion_x,t)
    PINN2 = P2model(tf.concat([intercepcion_x, t], axis=1))  # (intercepcion_x) se evalua con este valor de (x) porque aqui es donde se interceptan, y las dos PINN deben evaluarse en su intersección para ver la diferencia entre ellas
    return (tf.reduce_mean((PINN1- PINN2)**2))      # (t) se evalua en este valor porque siguen teniendo el mismo (t), este no se partio, esta dividida por el espacio


#tercer paso, generamos la data de la EDP (SEGUN LA ECUACION DEL CALOR) y definimos el entrenamiento
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#(1)Defino la data
#...........................................................................................................................................................................................................................................................................

def funcion_valor_inicial(x):
    return 5.0 * (1.0 - tf.pow(x,2))  # MI FUNCIÓN DE PROBLEMA DE VALOR INICIAL


# Coeficiente de difusión del calor (parametro físico)
gamma = 1
beta = 2
 #PINN LEONARDO MIRAR
 
T_final = 0.3170 # Este es el tiempo estimado de la explosión para u^2 en el experimento númerico

#Aqui definimos la cantidad de puntos para la PINN
Puntos_dom = 5000
Puntos_ic= 500
Puntos_bn = 250

# Para los datos de la PDE, x ∈ (-1,1) t>0, como de todos modos toca definir un intervalo final en t, se dejara hasta el tiempo donde se supone que explota
# Eso segun el experimento númerico
x = tf.random.uniform((Puntos_dom, 1), minval=-1.0, maxval=1.0, dtype=tf.float64)
if puntero_condicion_adaptattiva[0]=='S':
    t = T_final*(1-tf.random.uniform((Puntos_dom, 1),minval = 0.0 , maxval = 1.0, dtype=tf.float64)**beta) #PESOS PONDERADOS ADAPTATIVAMENTE (2)
else:
    t = tf.random.uniform((Puntos_dom, 1), minval=0.0,    maxval=T_final,  dtype=tf.float64)           #t DEFINIDO NORMALMENTE           (1)

  
 
# Para los datos en la condición inicial, x_in esta igual de definido que en x, es decir x_in ∈ (-1,1)
# para t_in todos los puntos generados son cero, puesto que comienzo en u(x,0), es por eso t=0
x_in = tf.random.uniform((Puntos_ic, 1), minval=-1.0, maxval=1.0, dtype=tf.float64)
t_in = tf.zeros_like(x_in, dtype=tf.float64) # en u(x_in,0) genere la cantidad de puntos donde esta definido x_in
u0_func = funcion_valor_inicial(x_in) 

#Left, Righ
# Para los datos en la fronetra
t_bn  = tf.random.uniform((Puntos_bn, 1), minval=0.0, maxval=T_final, dtype=tf.float64) # porque como cuando definifimos x_in arriba, el x de la condición inicial esta (-1,1), el t_bn esta creciendo, es decir ∈ t>0 (leer el articulo para comprenderlo)
x_bnL = tf.ones_like(t_bn, dtype=tf.float64) * -1.0 # en u(-1,t_bn) genere la cantidad de puntos de t_bn para la frontera izquierda 
x_bnR = tf.ones_like(t_bn, dtype=tf.float64) * 1.0#  en u(1,t_bn) genere la cantidad de puntos de t_bn para la frontera derecha             
u_bn_func  = tf.zeros_like(t_bn, dtype=tf.float64) # u(-1,t_bn)=u(1,t_bn)=0 para la función en la frontera, genere al comienzo, es decir en t=0, los puntos generados


x_line = tf.linspace(tf.constant(-1.0, tf.float64), tf.constant(1.0, tf.float64), Puntos_ic)
u0_line = funcion_valor_inicial(tf.reshape(x_line, (-1,1)))
distribucion_dominio_y_condicion_inicial_GRAF(x,t,x_in,t_in,u0_func,x_line,u0_line,x_bnL,x_bnR,t_bn,T_final)
#NOTA: para entender como se definio esta data, se debe de leer el articulo de la explosión númerica del blow up y ver como esta defnido el (x,t) para el dominio de la PDE, en las condiciones iniciales y la frontera
#Por el momento, el x_in y el t_bn dependen del "x" y "t" del dominio porque tienen las mismas condiciones, pero en el notebook se ven que los generan aparte
#solamente para que pueda generar una cantidad de puntos independientes, osea una cantidad determinada adentro ene el dominio, otra cantidad en condiciones iniciales y otra en la frontera
#hay que preguntar si necesitamos la mimsa cantidad generada de puntos en todos lados o o, y dependiendo de eso cambairemos la dependenca de x_in y t_bn despues


#____________________________________________________________________________________________________________________________________
#DOMINIOS XPINN
#Datos Dominio

x_mid = 0.0 # porque en x=0 se arte el dominio
x1 = tf.random.uniform((Puntos_dom, 1), minval=-1.0, maxval=x_mid, dtype=tf.float64)
x2 = tf.random.uniform((Puntos_dom, 1), minval=x_mid, maxval=1.0, dtype=tf.float64)                      #-1<x1<0 0<x2<1
t = tf.random.uniform((Puntos_dom, 1), minval=0.0,    maxval=T_final,  dtype=tf.float64)               # t>0

#t = T_final*(1-tf.random.uniform((Puntos_dom, 1),minval = 0.0 , maxval = 1.0, dtype=tf.float64)**beta)               # t>0

#Datos Cond Ini


x1_in = tf.random.uniform((Puntos_ic, 1), minval=-1.0, maxval=x_mid, dtype=tf.float64)
t1_in = tf.zeros_like(x_in, dtype=tf.float64)
u0_func1 = funcion_valor_inicial(x1_in)

x2_in = tf.random.uniform((Puntos_ic, 1), minval=x_mid, maxval=1.0, dtype=tf.float64)
t2_in = tf.zeros_like(x_in, dtype=tf.float64)
u0_func2 = funcion_valor_inicial(x2_in) 



#Frontera

t_bn  = tf.random.uniform((Puntos_bn, 1), minval=0.0, maxval=T_final, dtype=tf.float64) 
x_bnL = tf.ones_like(t_bn, dtype=tf.float64) * -1.0
x_bnM = tf.ones_like(t_bn, dtype=tf.float64) * x_mid # porque hay se prte el dominio, t_bn declara en todos, comienzan en 0 y van la linea vertical de las fronteras, y van hasta el T_final, en este caso L, M y R creceran hasta el t de la explosión
x_bnR = tf.ones_like(t_bn, dtype=tf.float64) * 1.0
u_bn_func  = tf.zeros_like(t_bn, dtype=tf.float64) 


distribucion_dominio_y_condicion_inicial_GRAF(x1,t,x1_in,t1_in,u0_func1,x_line,u0_line,x_bnL,x_bnM,t_bn,T_final)
distribucion_dominio_y_condicion_inicial_GRAF(x2,t,x2_in,t2_in,u0_func2,x_line,u0_line,x_bnM,x_bnR,t_bn,T_final)

epsilon = 1e-3 #para cuando T_final=t, y no divida por 0, un valor demasiado minusculo para que no altere lo demas

w_ti = 1.0 / (T_final - t+ epsilon)**gamma 


#...........................................................................................................................................................................................................................................................................

#(2)defino mi modelo (cuantas capas y que optimizador quiero, usualmente es adam, y el número de epocas que se entrenara)

"""
model = PINN([2,30, 30, 30, 30, 1])
_ = model(tf.zeros((1,2), dtype=tf.float64))
"""
#AQUI CREO MIS DOS PINNs
P1model = PINN([2,30, 30, 30, 30, 1])
P2model = PINN([2,30, 30, 30, 30, 1])

epochs = 1000
lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=[int(0.33*epochs), int(0.66*epochs)],
    values=[tf.constant(1e-2, tf.float64), tf.constant(1e-3, tf.float64), tf.constant(1e-4, tf.float64)]
) 
optimizer1 = tf.keras.optimizers.Adam(learning_rate=lr_sched) #(4)5e-4
optimizer2 = tf.keras.optimizers.Adam(learning_rate=lr_sched) #(4)5e-4

#...........................................................................................................................................................................................................................................................................


#(3)ENTRENAMIENTO! (ACTUALIZACIÓN, se adiciona en el código la forma de guardar los mejores pesos entrenados)
best_epoch = 0  #guardamos la epoca donde mejor fue entrenado el modelo
#best_loss = 50000000.0 #guardamos el función de perdida más baja que dio, para saber cual fue el mejor entrenamiento, comenzamos en un valor muy alto para hacer comparación

w_pde = 0.3  # Prioriza la pérdida de la EDP
w_ic = 0.4  # Para obtener Blow Up es muy importante, depende de la condición inicial, más no de la frontera
w_bc = 0.3    

"""
P1History_loss_1 = []
P1History_loss_2 = []
P1History_loss_3 = []
P1History_total_loss= []

P2History_loss_1 = []
P2History_loss_2 = []
P2History_loss_3 = []
P2History_total_loss= []
"""


#Capturar tiempo
start = time.time() 
for epoch in range(1, epochs+1):

    with tf.GradientTape(persistent=True) as tape:

        loss_interfaz_x = Interfaz_espacio_loss(P1model,P2model,x,t,x_mid)

        #funciones de perdidida
        P1loss_pde = EDP_loss(P1model, x1, t,w_ti)
        P1loss_in = Valores_Iniciales_loss(P1model, x1_in, t1_in, u0_func1)
        P1loss_bn = Frontera_loss(P1model, x_bnL, x_bnM , t_bn, u_bn_func)

        PINN1_loss = w_pde*P1loss_pde + w_ic*P1loss_in + w_bc*P1loss_bn+loss_interfaz_x


        P2loss_pde = EDP_loss(P2model, x2, t,w_ti)
        P2loss_in = Valores_Iniciales_loss(P2model, x2_in, t2_in, u0_func2)
        P2loss_bn = Frontera_loss(P2model, x_bnM, x_bnR , t_bn, u_bn_func)

        PINN2_loss = w_pde*P2loss_pde + w_ic*P2loss_in + w_bc*P2loss_bn+loss_interfaz_x 

        #bacpropagation
        gradients_1 = tape.gradient(PINN1_loss, P1model.trainable_variables)
        optimizer1.apply_gradients(zip(gradients_1, P1model.trainable_variables))
        loss_P1 = PINN1_loss.numpy()
       
        
        gradients_2 = tape.gradient(PINN2_loss, P2model.trainable_variables)
        optimizer2.apply_gradients(zip(gradients_2, P2model.trainable_variables))
        loss_P2 = PINN2_loss.numpy()  

        """
        P1History_loss_1.append(P1loss_pde)
        P1History_loss_2.append(P1loss_in)
        P1History_loss_3.append(P1loss_bn)
        P1History_total_loss.append(PINN1_loss.numpy())  # Almacenar la pérdida total

        P2History_loss_1.append(P1loss_pde)
        P2History_loss_2.append(P1loss_in)
        P2History_loss_3.append(P1loss_bn)
        P2History_total_loss.append(PINN1_loss.numpy())  # Almacenar la pérdida total
        """
        
        
        
        
        
        
    """
    if  loss < best_loss: #si encontro una función de perdida donde aprendio mejor (redujo más), guarde cuanto fue, en que época y guarde esos pesos que fueron los mejroes entrenados
        best_loss = loss
        best_epoch = epoch
        model.save_weights("mejores_pesos_pinn.weights.h5")        #(P)
    """    


    
    if epoch%100==0:
        P1xt = tf.concat([x1, t], axis=1)
        P2xt = tf.concat([x2, t], axis=1)
        u1 = P1model(P1xt)
        u2 = P2model(P2xt)
        u_max_1 = tf.reduce_max(tf.abs(u1)).numpy()
        u_max_2 = tf.reduce_max(tf.abs(u2)).numpy()
        print(f"epoca {epoch} | sol_1_max: {u_max_1:.5f} | sol_2_max: {u_max_2:.5f}")
     

    
    print(f"Epoch {epoch}/{epochs},Loss_PINN1: {loss_P1:.5f}, Loss_PINN2: {loss_P2:.5f} ") #{loss:.5f}

end = time.time()
print("Tiempo de ejecución:", end - start, "segundos")
#print("mejor epoca de entrenamiento:", best_epoch)
#print("valor de la PINN_loss:", best_loss )

#---------------------------------------------------------------------------------------------
#PESOS
#model.load_weights("mejores_pesos_pinn.weights.h5") #cargue los mejores pesos del entrenamiento  (P)
#---------------------------------------------------------------------------------------------


#cuarto paso, mostramos vizualisaciones de los datos
#---------------------------------------------------------------------------------------------
#oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
x_vis1 = tf.linspace(-1.0, 0.0, 150)
x_vis2 = tf.linspace(0.0, 1.0, 150)
t_vis = tf.linspace(0.0, T_final, 250)
# ESTO DE ACA ES LO QUE PERMITE VIZUALIZAR BIEN LA MALLA 23/03/2026, 
# * Uno de los errores que se estaba cometiendo al querer visualizar la solución de u_φ era que antes se mandaba todos los millones de puntos a graficar
# Lo cual no era necesario, con unos pocos y estableciendo los (x) y (t) definidos de la EDP ya sale bien dibujado
# * Antes aunque tuviera millones de puntos en (x) y (t), si dibujaba fluida ¿Por Qué? Porque estaban uniformemente espaciados
# Lo cual para el computador era más facil de graficar, pero estaban mal definidos, porque las variables debian ser uniformemente aleatorias
# Ahora, como estan definidas uniformemente aleatorias, graficar era un caos!!, entonces unicamente (x_vis,t_vis) si seran uniformemente y con ellas visualizare y dibujare la grafica
# ya que ellas computacionalmente si son faciles de graficas, y como solo las estoy usando para visualizar, más no para entrenar, no afecta en nada a la verdadera solución de u_φ

# 1. Crear la malla con tf.meshgrid
X1, T = tf.meshgrid(x_vis1, t_vis)  # X representa las posiciones espaciales, T los tiempos
X2, T = tf.meshgrid(x_vis2, t_vis)  # X representa las posiciones espaciales, T los tiempos
# 2. Aplanar para formar los inputs al modelo
X1_flat = tf.reshape(X1, [-1])
X2_flat = tf.reshape(X2, [-1])   
T_flat = tf.reshape(T, [-1])
input_tensor1 = tf.stack([X1_flat, T_flat], axis=1)  # (N, 2) donde N = 10000
input_tensor2 = tf.stack([X2_flat, T_flat], axis=1)  # (N, 2) donde N = 10000
# 3. Evaluar la PINN
U_pred_1 = P1model(input_tensor1)  # u(x,t) predicho por la red
U_pred_2 = P2model(input_tensor2)  # u(x,t) predicho por la red

U_pred_1 = tf.reshape(U_pred_1, X1.shape)  # Volver a la forma (100,100)
U_pred_2 = tf.reshape(U_pred_2, X2.shape)  # Volver a la forma (100,100)

# 4. Convertir a NumPy para graficar
X2_np = X1.numpy()
X1_np = X2.numpy()
T_np = T.numpy()


#La solución de la Ecuación Diferencial dada por la PINN se denoratara por: ---(u_φ)---

  




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

# NUEVOS GRAFICOS CONCATENADOS

asintota = 1200

#funciones_de_perdida_GRAF(x,asintota,History_loss_1,History_loss_2,History_loss_3,History_total_loss)


u_φ = U_pred_1.numpy()

mapa_de_calor_y_grafico_3D_para_la_PINN_GRAF(u_φ,X1_np,T_np)

u_φ = U_pred_2.numpy()

mapa_de_calor_y_grafico_3D_para_la_PINN_GRAF(u_φ,X2_np,T_np)


#PINN_vs_Diferencias_finitas_GRAF(u_φ,X_np,T_np)

#NOTAS

"""
versión 1

* XPINN partido por x, x1 en [-1,0] y x2 en [0,1]

"""


