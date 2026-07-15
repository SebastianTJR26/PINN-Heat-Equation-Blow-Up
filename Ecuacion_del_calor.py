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

#PUNTERO: Menjear arreglos es lo más cercano a trabajar con las direcciones de memoria en python, como necesito cambiar la función de activación
#Para poder ver como les va a los demas experimentos con otras funciones, se crea una arreglo que contiene solo un dato tipo string en la posición 0 con el nombre de una función de activación
#Y como es un arreglo, cuando cambie la fcn de activación para una arquitectura, se que la red lo implementara porque "activation" esta hacia la dirección del string de l fcn de activación
puntero_fcn_activacion = ['tanh'] 
from keras.saving import register_keras_serializable

condicion = input("Activar muestreo adaptativo (S/n):")
if condicion=='S':
    puntero_condicion_adaptattiva = ['S']
else:
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
        self.hidden_layers = [tf.keras.layers.Dense(layer, activation=puntero_fcn_activacion[0]) for layer in layers[:-1]]

        # Capa de salida con activación None
        self.output_layer = tf.keras.layers.Dense(layers[-1], activation=None)

    def call(self, inputs): # Forward propagation
        x = inputs[:, 0:1]  # Extrae la primera columna (posición x)
        τ = inputs[:, 1:2]  # Extrae la segunda columna (tiempo τ) 
        # Separa las entradas en x y τ, ya que la EDP que queremos hallar está en función de u_φ(x,τ)
        # siendo "x" posición y "τ" el tiempo
        # Son dos entradas porque esta EDP está en función de "x" y "τ"

        CAPA_EDP = tf.concat([x, τ], axis=1)  
        # Concatena "x" y "τ". Estos son los DATOS de la EDP, 
        # se concatenan porque si no, tocaría hacer un entrenamiento separado para "x" y otro para "τ"

        for layer in self.hidden_layers:  
            CAPA_EDP = layer(CAPA_EDP)  # Aplica cada capa densa

        CAPA_EDP = self.output_layer(CAPA_EDP)  
        # Aplica la capa de salida sin activación, ya que estamos modelando u_φ(x,τ)
            
        return CAPA_EDP  # Devuelve la salida de la red
  
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#segundo paso ,La función de pérdida es el núcleo de las PINNs. Combina tres componentes:
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#(1) Pérdida de la PDE: Mide cuánto se desvía la solución de la red neuronal de la ecuación diferencial. ∥∂u/∂t-α*∂²u/∂x²∥²
#...........................................................................................................................................................................................................................................................................
#antes, las condiciones iniciales y la frontera estaban en stack, no concatenadas, ahora ya si estan concatenadas

#La solución de la Ecuación Diferencial dada por la PINN se denoratara por: ---(u_φ)---

def EDP_loss(model, x, τ, T_blow_up, epsilon, gamma):
    # tf.GradientTape es una herramienta de TensorFlow para calcular derivadas (gradientes) de operaciones con respecto a sus variables.
    # with asegura que todas las operaciones dentro de él sean "grabadas" por GradientTape, para que luego se puedan calcular sus gradientes.
    # El parámetro persistent=True permite que la cinta (tape) sea reutilizable para calcular múltiples gradientes. 
    # Sin esto, la cinta se borraría después de calcular el primer gradiente.
    # tf.reduce_mean promedia estos errores para obtener una pérdida escalar.
    # tape.watch() le dice a TensorFlow que vigile o registre las operaciones que dependen de variables. 
    # Esto es necesario cuando variable no es un tensor entrenable (es decir, no es un parámetro de la red neuronal, 
    # como los pesos o biases), pero aún así necesitas calcular derivadas con respecto a esa variable.

    w_ti = 1.0 / tf.pow(T_blow_up - τ + epsilon, gamma) #como esto no se esta derivando, no es necesario su verificación

    
    with tf.GradientTape(persistent=True) as tape:

        tape.watch(x)
        tape.watch(τ)

        # Combina los tensores x y t a lo largo del eje 1, en pocas palabras, crea la función u(x,τ).
        # OJO: ESTA NO ES LA SOLUCIÓN ANALÍTICA O NUMÉRICA, es una (u) formada por la red.
        # Cambia u_φ -> u_pred en caso de confusión.
        u_φ = model(tf.concat([tf.reshape(x, (-1, 1)), tf.reshape(τ, (-1, 1))], axis=1))

        # Diferenciación automática, calcula la derivada de la función u con respecto a t
        du_dτ = tape.gradient(u_φ, τ)  

        # Diferenciación automática, calcula la derivada de la función u con respecto a x
        du_dx = tape.gradient(u_φ, x)  

        # Diferenciación automática, calcula la segunda derivada de la función u con respecto a x
        d2u_dx2 = tape.gradient(du_dx, x)  

        # Las derivadas deben estar dentro del tape, porque TensorFlow está rastreando si se hace bien.
        # Si están fuera, podrían dar None.

    residuo_edp = du_dτ - T_blow_up*tf.math.exp(-τ)*( d2u_dx2 + tf.pow(u_φ,2) )  #nuevo residuo del cambio de variable  u_φ(x,t)->u_φ(x,τ) 
    
    if puntero_condicion_adaptattiva[0]=='S':
        #RETURN 2 CON PONDERACIÓN ADAPTATIVA
        #return tf.reduce_mean((residuo_edp)**2) 
        #return tf.reduce_sum(w_ti * tf.reduce_mean((residuo_edp)**2)) / tf.reduce_sum(w_ti) #-> (Σ w_ti * ||R||²)/ Σw_ti ESTA BIEN,  SI CUMPLE BIEN LA DEFINICIÓN
        #return tf.reduce_sum(w_ti * tf.square(residuo_edp)) / tf.reduce_sum(w_ti) #-> (Σ w_ti * √R)/ Σw_ti ESTA MAL, NO ESTA SACANDO LA NORMA AL CUADRADO

        # Da igual el residuo que escojas acá, para el cambio de variable noe sta funcinando bien el muestreo adaptativo, y menos los pesos w_ti
        # Pero recuerda aquivar uno

    else:
        #RETURN 1 SIN PONDERACIÓN ADAPTATIVA
        return tf.reduce_mean((residuo_edp)**2) 
    
# tf.reduce_mean((residuo_edp)**2) = tf.square(residuo_edp)

# Otra forma de hacerlo:
# u = model(tf.stack([x, t], axis=1))  # tf.stack([x, t], axis=1) y tape.watch([x, t]) EN REALDIAD ESTO ESTA MAL, DESDE QUE CAMBIE A (contac) SI MEJORO EL RESULTSADO, TOCA VER QUE HACE (stack) Y VER ESPECIFICAMENTE EN QUE SE DIFERENCIAN 

#...........................................................................................................................................................................................................................................................................
 
#(2) Cálculo con los valores inicales de la EDP para su función de perfida . ∥u(x,0)−u(x)∥² Mide cuanto se desvía la solución de la red neuronal de las condiciones iniciales.
#...........................................................................................................................................................................................................................................................................
def Valores_Iniciales_loss(model, x_in, τ_in, u0_func):
    #tape.watch([x, t_initial, u_initial])  # ¡Innecesario! ya que no se hara ninguna operación de diferencia finita, esto solo es cuando uno hace gradiente a cosas no declaradas con tensorflow, como numpy

    # Predecir u(x, t_initial) usando la red neuronal
    u_φ0 = model(tf.concat([x_in, τ_in], axis=1))
    # Calcular la pérdida: (u - u_initial)²
    return (tf.reduce_mean((u_φ0- u0_func)**2))

#...........................................................................................................................................................................................................................................................................


#(3) Perdida en la frontera: Mide cuánto se desvía la solución de la red neuronal de las condiciones de frontera. ∥u(0,t)-u_boundary(t)∥²+∥u(L,t)-u_boundary(t)∥²
#...........................................................................................................................................................................................................................................................................
def Frontera_loss(model, x_bnL, x_bnR , τ_bn, u_bn_func):
    u_φL = model(tf.concat([x_bnL, τ_bn], axis=1))
    u_φR = model(tf.concat([x_bnR, τ_bn], axis=1))
    
    loss_L = tf.reduce_mean((u_φL - u_bn_func) ** 2)
    loss_R = tf.reduce_mean((u_φR - u_bn_func) ** 2)
    return loss_L+loss_R
    #TANTO LA FUNCION DE PERDIDA DE LA FRONTERA COMO LOS VALORES INICIALES ESTABAN EN tf.stack Y NO tf.contac, TAL VEZ ESO INFLUYERA EN EL ERROR DEL TENSOR DIFERENTE
#...........................................................................................................................................................................................................................................................................



#tercer paso, generamos la data de la EDP (SEGUN LA ECUACION DEL CALOR) y definimos el entrenamiento
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#(1)Defino la data
#...........................................................................................................................................................................................................................................................................

def funcion_valor_inicial(x):
    return 5.0 * (1.0 - tf.pow(x,2))  # MI FUNCIÓN DE PROBLEMA DE VALOR INICIAL


# Constante de la ponderación del muestreo adaptativo
gamma = tf.constant(2.0, dtype=tf.float64)
 
 
T_blow_up = 0.3170  #CORRECIÓN: T_final nunca debio cambiar el valor de 0.3170, ya que el t0 dela artitulo es el tiempo finito de exploción y por ende la ecuación cuando se hizoo el cambio de variable
                    #Debe de entender que viene de una explosión finitia y en algun lado debe estar porque ambas ecuaciones estan correlacionadas, entones aunque tao vaya a infinito, T_final debio ser siempre 0.3170, ahora llamado T_blow_up

#Aqui definimos la cantidad de puntos para la PINN
Puntos_dom = 10000
Puntos_ic= 1000
Puntos_bn = 500
# x ∈ (-1,1), τ>0 los imtervalos según la EDP con cambio de variable
minvalx = -1.0
maxvalx = 1.0
minvalτ = 0.0
maxvalτ = 2.0

# Para los datos de la PDE, x ∈ (-1,1) τ>0, como de todos modos toca definir un intervalo final en τ, se dejara hasta el tiempo donde se supone que explota
# Eso segun el experimento númerico
x = tf.random.uniform((Puntos_dom, 1), minval=minvalx, maxval=maxvalx, dtype=tf.float64)
if puntero_condicion_adaptattiva[0]=='S':
    τ = maxvalτ*(1-tf.random.uniform((Puntos_dom, 1),minval = 0.0 , maxval = 1.0, dtype=tf.float64)**2) #PESOS PONDERADOS ADAPTATIVAMENTE (2)
    #τ = maxvalτ*(1-tf.random.uniform((Puntos_dom, 1),minval = minvalτ , maxval = maxvalτ, dtype=tf.float64)**2)
    #ESTA MAL ESTA DEFINICIÓN!! YA QUE LA ORIGINAL ES : T_final*(1-Uniform(0,1)^B), y esta va de (minvalτ,maxvalτ), mientras que maxvalτ=1 se mantiene bien, pero cuando cambiemos ha maxvalτ>1 se rompe, por eso siempre debe ser (0,1)
else:
    τ = tf.random.uniform((Puntos_dom, 1), minval=minvalτ,  maxval=maxvalτ,  dtype=tf.float64)           #τ DEFINIDO NORMALMENTE           (1)   
    

# Para los datos en la condición inicial, x_in esta igual de definido que en x, es decir x_in ∈ (-1,1)
# para τ_in todos los puntos generados son cero, puesto que comienzo en u(x,0), es por eso t=0
x_in = tf.random.uniform((Puntos_ic, 1), minval=minvalx, maxval=maxvalx, dtype=tf.float64)
τ_in = tf.zeros_like(x_in, dtype=tf.float64) # en u(x_in,0) genere la cantidad de puntos donde esta definido x_in
u0_func = funcion_valor_inicial(x_in) 

#Left, Righ
# Para los datos en la frontera
τ_bn  = tf.random.uniform((Puntos_bn, 1), minval=minvalτ, maxval=maxvalτ, dtype=tf.float64) # porque como cuando definifimos x_in arriba, el x de la condición inicial esta (-1,1), el t_bn esta creciendo, es decir ∈ t>0 (leer el articulo para comprenderlo)
x_bnL = tf.ones_like(τ_bn, dtype=tf.float64) * minvalx#*-1  en u(-1,τ_bn) genere la cantidad de puntos de t_bn para la frontera izquierda 
x_bnR = tf.ones_like(τ_bn, dtype=tf.float64) * maxvalx#*1  en u(1,τ_bn) genere la cantidad de puntos de t_bn para la frontera derecha             
u_bn_func  = tf.zeros_like(τ_bn, dtype=tf.float64) # u(-1,τ_bn)=u(1,τ_bn)=0 para la función en la frontera, genere al comienzo, es decir en t=0, los puntos generados

#NOTA: para entender como se definio esta data, se debe de leer el articulo de la explosión númerica del blow up y ver como esta defnido el (x,τ) para el dominio de la PDE, en las condiciones iniciales y la frontera
#Por el momento, el x_in y el τ_bn dependen del "x" y "τ" del dominio porque tienen las mismas condiciones, pero en el notebook se ven que los generan aparte
#solamente para que pueda generar una cantidad de puntos independientes, osea una cantidad determinada adentro ene el dominio, otra cantidad en condiciones iniciales y otra en la frontera
#hay que preguntar si necesitamos la mimsa cantidad generada de puntos en todos lados o o, y dependiendo de eso cambairemos la dependenca de x_in y τ_bn despues



x_line = tf.linspace(tf.constant(minvalx, tf.float64), tf.constant(maxvalx, tf.float64), Puntos_ic)
u0_line = funcion_valor_inicial(tf.reshape(x_line, (-1,1)))
distribucion_dominio_y_condicion_inicial_GRAF(x,τ,minvalx,maxvalx,minvalτ,maxvalτ,x_in,τ_in,u0_func,x_line,u0_line,x_bnL,x_bnR,τ_bn,T_blow_up)

epsilon = 1e-3 #para cuando T_final=τ, y no divida por 0, un valor demasiado minusculo para que no altere lo demas




#...........................................................................................................................................................................................................................................................................

#(2)defino mi modelo (cuantas capas y que optimizador quiero, usualmente es adam, y el número de epocas que se entrenara)
#model = PINN([20, 20, 20, 1])#la ultima capa siemrpe debe ser una neurona!! (1)
#model = PINN([50,25, 1]) #mejor explosión (2)
model = PINN([2,60, 60, 60, 60, 1])
#model = PINN([10, 15, 20, 25,30, 1])
#model = PINN([100,50,25, 1]) #sin el 25 da más temperatura, pero menos que el modelo anterior, y alborota más la fcn de perdida (3)
#model = PINN([50,1])
epochs = 1000
lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=[int(0.33*epochs), int(0.66*epochs)],
    values=[tf.constant(1e-2, tf.float64), tf.constant(1e-3, tf.float64), tf.constant(1e-4, tf.float64)]
) 
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_sched) #(4)5e-4

#...........................................................................................................................................................................................................................................................................


#(3)ENTRENAMIENTO! (ACTUALIZACIÓN, se adiciona en el código la forma de guardar los mejores pesos entrenados)
best_epoch = 0  #guardamos la epoca donde mejor fue entrenado el modelo
#best_loss = 50000000.0 #guardamos el función de perdida más baja que dio, para saber cual fue el mejor entrenamiento, comenzamos en un valor muy alto para hacer comparación

w_pde = 0.3  # Prioriza la pérdida de la EDP
w_ic = 0.4  # Para obtener Blow Up es muy importante, depende de la condición inicial, más no de la frontera
w_bc = 0.3    

History_loss_1 = []
History_loss_2 = []
History_loss_3 = []
History_total_loss= []


#Capturar tiempo
start = time.time() 
for epoch in range(1, epochs+1):

    with tf.GradientTape() as tape:

        loss_1 = EDP_loss(model, x, τ, T_blow_up, epsilon, gamma)

        loss_2 = Valores_Iniciales_loss(model, x_in, τ_in, u0_func)

        loss_3 = Frontera_loss(model, x_bnL, x_bnR , τ_bn, u_bn_func)

        History_loss_1.append(loss_1)

        History_loss_2.append(loss_2)

        History_loss_3.append(loss_3)

        PINN_loss = w_pde*loss_1 + w_ic*loss_2 + w_bc*loss_3

        History_total_loss.append(PINN_loss.numpy())  # Almacenar la pérdida total

        gradients = tape.gradient(PINN_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        loss = PINN_loss.numpy()
    """
    if  loss < best_loss: #si encontro una función de perdida donde aprendio mejor (redujo más), guarde cuanto fue, en que época y guarde esos pesos que fueron los mejroes entrenados
        best_loss = loss
        best_epoch = epoch
        model.save_weights("mejores_pesos_pinn.weights.h5")        #(P)
    """    

    if epoch%100==0:
        xτ = tf.concat([
        tf.reshape(x, (-1, 1)),
        tf.reshape(τ, (-1, 1))
        ], axis=1)
        u_pred = model(xτ)                                                  #al parecer, esto me esta diciendo en que época obtuve mi blow-up, tengo que ver si hubo una variación brusca
        intensidad= tf.reduce_max(tf.abs(u_pred)).numpy()
        print("u_max en la epoca ", epoch," :",intensidad)

        

    
    print(f"Epoch {epoch}/{epochs},Loss_PINN: {loss}") #{loss:.5f}

end = time.time()
print("Tiempo de ejecución:", end - start, "segundos")
print("mejor epoca de entrenamiento:", best_epoch)
#print("valor de la PINN_loss:", best_loss )

#---------------------------------------------------------------------------------------------
#PESOS
#model.load_weights("mejores_pesos_pinn.weights.h5") #cargue los mejores pesos del entrenamiento  (P)
#---------------------------------------------------------------------------------------------


#cuarto paso, mostramos vizualisaciones de los datos
#---------------------------------------------------------------------------------------------
#oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
x_vis = tf.linspace(minvalx, maxvalx, 100)#tf.linspace(minvalx, maxvalτ, 100) mira como estaba definida, TENIAS maxvalτ EN EL TENSOR DE x !! XD
τ_vis = tf.linspace(minvalτ, maxvalτ, 250)
# ESTO DE ACA ES LO QUE PERMITE VIZUALIZAR BIEN LA MALLA 23/03/2026, 
# * Uno de los errores que se estaba cometiendo al querer visualizar la solución de u_φ era que antes se mandaba todos los millones de puntos a graficar
# Lo cual no era necesario, con unos pocos y estableciendo los (x) y (t) definidos de la EDP ya sale bien dibujado
# * Antes aunque tuviera millones de puntos en (x) y (t), si dibujaba fluida ¿Por Qué? Porque estaban uniformemente espaciados
# Lo cual para el computador era más facil de graficar, pero estaban mal definidos, porque las variables debian ser uniformemente aleatorias
# Ahora, como estan definidas uniformemente aleatorias, graficar era un caos!!, entonces unicamente (x_vis,t_vis) si seran uniformemente y con ellas visualizare y dibujare la grafica
# ya que ellas computacionalmente si son faciles de graficas, y como solo las estoy usando para visualizar, más no para entrenar, no afecta en nada a la verdadera solución de u_φ

# 1. Crear la malla con tf.meshgrid
X, T = tf.meshgrid(x_vis, τ_vis)  # X representa las posiciones espaciales, T los tiempos

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
u_φ = U_pred.numpy()

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

asintota = 400

funciones_de_perdida_GRAF(x,asintota,History_loss_1,History_loss_2,History_loss_3,History_total_loss)


mapa_de_calor_y_grafico_3D_para_la_PINN_GRAF(u_φ,X_np,T_np)


PINN_vs_Diferencias_finitas_GRAF(u_φ,X_np,T_np)
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



"""
versión 18

*Con Nx=Nt=200 sigue alcanzando una temperatura de 5.7, pero la fcn de perdida disminuto más, la total, aunque en los valores iniciales enloquecio un poco
*El n/(1+tf.norm(u, ord=2) definido tanto en esta versión como en la 17 era n=100, se me olvido incluir ese pequeño comentario

"""


"""
versión 19

*Con n/(1+tf.norm(u, ord=2) n=1000 llegamos a 6!, la fcn de perdida baja considerablemente,s e ve mejor la total, pero la de los valores inciales esta creciendo mucho!

"""



"""
versión fantasma

*Se olvida registrar esta abotación, en el git dodne aparece Nx y Nt en 400 volvio aumentar la tempereatura, se puede encontrar antes de la versión 20



"""

"""
versión 20

*Se han incluido los cambios sugeridos por el profesor Leonardo
*(1) las garficas asintoticas que miran als ultimas 1200 épcoas de las fcn's de perdida
*(2) Los pesos de las funciones de perdida de (bc) y (ic) se dejaron con w=0 y solo estamos contando la de la (pde)
*Por alguna razón, al hacer ese cambio las fcn en w=0 (wights) cuando se esta optimizando el problema, las graficas de ellas cambiaron al final, solo cuando w!= 0 no era asi
*(3) con la norma la temperatura lleg a 9 !!! solo contando la fcn loss de pde
*(4) sin la norma no ayuda mucho, antes baja la temperatura demasiado, me toco cambairle los learning rate porque estaba la fcn de eprda llegando a valores negativos
*(5) No me gusta como se hizo el grafico de la distribución de puntos, fue generado por ChatGPT, debo de hacer una versión propia y entenderlo porque no esta funcionando bien
*(6) se incluyo el código del profesor Leoanrdo en otro .py, sigo estudiandolo como funciona, auqnue entinedo la forma que reparten los datos

"""

"""
versión 21

*Antes de esta versión, se guardaron unos cambios en la misma fecha que no se registraron, entre ellos fue dejar en solo pleteo todas las funcciones de perdida y sus graficos asintotios ultimas n épocas
*Se tenia entrenado a 3000 épocas con ultima capa relu, la temperatura era 7.2, de hay para adelante por más modificaciones que se hicieran no ayudo mucho
*Ahora el entrenamiento se extendio a 6500 épocas,, la ultima capa se desactivo y Nx y Nt en ambas versiones estaban en 400, la temperatura llego a 9.26

"""


"""
versión 22
*Esta es una versión en donde se cambio todas las capaz de tanh por relu, para ver si una fcn de activación no acotada mejoraba el crecimiento de la temperatura, resultado no fue asi :/

"""



"""
versión 23 NO ES TODAVIA EL BLOW UP, PERO ES LA VERSIÓN COMO FINAL DE ESTE CÓDIGO DE MOMENTO

*Este fue el mejor resultado obtenido con la PINN en la temperatura, donde todas las capaz usan activación tanh a pesar de ser acotada, y la salida sin activación
*Se incluyeron los códigos modificados tomados de los commits anteriores y se hicieron sus ramas alternas
_________________________________________________________________________________________________________________________________________________________________________________________________________________________
*llamado 2, que esta en la versión 2, commit (EDP op no lineal) se incluyo lo siguiente:
-Ahora tiene la grafica de las 4 funciones de de perdida, las astintoticas, el mapa de calor y el grafico 3D en color jet en un solo plot

_________________________________________________________________________________________________________________________________________________________________________________________________________________________
*llamado 3, que esta alrededor de la versión 16, commit (Blow Up con los parametros del articulo y modificaciones de Leonardo (Tenemos mejoria)) se incluyo lo siguiente:
-Ahora tiene la grafica de las 4 funciones de de perdida, las astintoticas, el mapa de calor y el grafico 3D en color jet en un solo plot

_________________________________________________________________________________________________________________________________________________________________________________________________________________________
*llamado 4, que esta en la versión 17, commit (Nx,Nt 200 temp 5.7 3000 epochs) se incluyo lo siguiente:
-Ahora tiene la grafica de las 4 funciones de de perdida, las astintoticas, el mapa de calor y el grafico 3D en color jet en un solo plot

_________________________________________________________________________________________________________________________________________________________________________________________________________________________
*llamado 1, que esta en la versión 1, commit (EDP original) QUE ES LA LINEA QUE TIENE LOS CAMBIOS MÁS IMPORTANTES se incluyo lo siguiente:

VERSIÓN ALTERNATIVA UNIFORME
-Ahora tiene la grafica de las 4 funciones de de perdida, las astintoticas, el mapa de calor y se incluyo un grafico 3D  en color jet en un solo plot
-Esta es la única versión que plote la colocación de los puntos del dominio y la condición inicial
-se incluyeron cosas posteriores como dejar la misma semilla para que el resultado de la EDP sea el mismo
-Tambien tiene la solución analítica

VERSIÓN ALTERNATIVA ALEATORIA
-Esta es la que incluye la versión donde los puntos se distribuyen aleatoriamente en el dominio y el gráfico que se debe estudiar, de todas las versiones alternas, ESTA ES LA ÚNICA QUE LA TIENE!!!
_________________________________________________________________________________________________________________________________________________________________________________________________________________________

*Una cosa que me percarte antes de la presentación de la propuesta, cuando Leonardo pidio cambiar la distribución a random en las variables (x,t), la EDP del calor se parecia más a la solución analitíca
*Pero me cambiaba muchos los resultados de todos mis experimentos y esta versión estable si le coloco ese cambio baja la temperatura, preocupa un poco y es mejor empezar a experimentar con una nueva versión de la PINN

"""




#AÑO 2026


"""
versión 24
*Por sugerencia del profe Leonardo, se dicidio modificar la condición inicial, en vez de ser 5(1-x^2) ahora es 10000(1-x^2), esto para ver si la red estaba reescalando despues de pasar a otra capa
*La red no re escalo y aumento a 40 la temperatura!!
*La condición de la norma con o sin ella no afecta en que crezca la red, pero con la norma la función de perdida de la EDP no se altera tanto, sin ella si empieza a tener a ver dificultades en el aprendizaje
*Esta vez con más tiempo de entrenamiento, 10000 épocas por ejemplo si se ve más crecimiento y forma de obtener más resultados, pro defecto aun lo dejamos en 6500
*la función de perdida esta extremadamente gigantezca, en posteriores modificaciones, sugerencias de Leonardo e ideas mias, ya tenemos pistas de como encontrar el Blow Up, más adelante anotaremos eso aqui y modificaremos el código, haber si funciona o no
"""



"""
versión 25

IMPLEMENTACIÓN PONDERADA HACIA EL BLOW UP EN CONSTRUCCIÓN

*1000/(1+tf.norm(u, ord=2)) ya no esta haciendo tanto efecto en la PINN

*gamma = tf.constant(1.0, dtype=tf.float32)
 T_final = 0.3170  Se agregaron estas constantes, este T final es el que explota en la ecuación númerica

* Este t es el que tira para el blow up (t = tf.convert_to_tensor(T_final*(1-np.linspace(0,1,Nt)**beta),dtype=tf.float32))

* en la red, se cambio u_pred o el solo u ha u_φ para indicar que esa es la solución que esta dada por la PINN 

*loss_1 = 0.0
loss_2 = 0.0
loss_3 = 0.0
loss_1 += EDP_loss(model, x, t, alpha,beta,T_final,epsilon,gamma)

loss_2 += Valores_Iniciales_loss(model, x_initial, t_initial, u_initial)

loss_3 += Frontera_loss(model, x_boundary, t_boundary, u_boundary)
segun la definiciones de las funcioens de eprdida, asi deberia ser, pero no me esta saliendo tan bien, estara propensa a una proxima revisión


*se hiozo la ponderación del blow up que mando Leoanrdo en el residuo de la PDE

*La norma se elimino cumpletamente

* los loss_n += no se implementaron al final, pero aun se debe 

"""



"""
versión 26

* Se hizo el cambio de tf.stack->tf.contac para las funciones de perdida de valores iniciales y frontera, tal vez eso hacia el problema de las dimensiones de los tensores y por eso cuando se colocaba este formato tf.random.uniform((Puntos_ic, 1), minval=-1.0, maxval=1.0, dtype=tf.float64) no dejaba
* La data ahora si esta bien definidoa, se pensaba mandarlas a funciones, pero no era tan necesario como se pensaba, salvo la función de problema de valor inicial, esa si se necesita tener en un metodo aparte
* Las graficas se movieron a un archivo llamado Graficos_PINN.py, y ahora los metodos de alla plotean todo
* La razón por la que los mapas de calor se dibujaban feo era por que mandaba dibujar millones de puntos, ahorita solo estoy mandando 250x100
* Aun esta pendiente la optimización del código
* Se hicierón cambios a Leonardo.py
* Se añadio un archivo de diferencias finitas para comparar (ES UN CÓDIGO MUY FACIL DE ENTENDER)
* Se movieron varios archivos antiguos.py a la carpeta de experimentos
* Anotar los cambios que hiciste para graficar las diferencias entre pinn y diferencias finitas antes de que se te olvide, como guardar las variables numericas y cosas asi (PENDIENTE)
* Ahora la PINN si se ve bien, y muestra que esta tratando de solucionar enserio la ecuación diferencial, todos los cambios incluso desde la versión anterior fueron los correctos
* Ya tengo super verificado que las funciones de perdida, estan totalmente bien, tal vez la nomenclatura aun puede variar en futuras versiones
* El tamaño de la solución de la pinn con la diferencias finitas puede ser diferente en X y T, debes mriar que los parametros sean iguales para comparar, sino, dara error de dimension como esto: ERROR (300,300) comparative with (250,100)
* El código esta mejor documentado y se puede entender que hace cada linea
* 05/04/2026 HUBO UN PROBLEMA QUE SE IDENTIFICO EN EL (return) de EDP_loss, Habian dos return cuando solo debia haber uno, y el que se estaba llamando estaba mal, porque no tenia encuenta los pesos adaptativos
* El segundo (return) que si era el que debia retornarse estaba mal definido, ya ahora si esta bien de acuerdo a la guia dada por el profe Leonardo
* Se dejaron para usar dos return, uno que es el de los pesos adaptativos en el T_final del blow up y el otro que no esta con pesos adaptativos
* FIJATE BIEN CUAL ESTAS LLAMANDO DEPENDIENDO DE LO QUE QUEIRAS (CON O SIN PESOS ADAPTATIVOS), ADEMAS FIJATE TAMBIEN QUE TIENE QUE ESTAR CON SU (t) correspondiente
* Se arreglarón algunos errores de ortografía
* No se tienen registros en la versión 25 de los códigos con los mapas de calor feos, nunca los guarde, pero con esta explicación se podrian recrear si uno quiere intentarlo, ademas esos mapas con poquitos puntos se demoraban en cargar
* Si por algún motivo vuelves a la (versión 25), se me olvido escribir el encabezado de la versión, pero comienza con (IMPLEMENTACIÓN PONDERADA HACIA EL BLOW UP EN CONSTRUCCIÓN)
"""

"""
versión 27 (nombre del commit: EXPERIMENTOS DE LA PINN QUE RESULVE LA EDP EN EXPLOSIÓN FINITA , CON Y SIN MUESTREO ADAPTATIVO (TABLAS) )

* Esta versión se creo unicamente para sacar los resultados de mi modelo por indicaciones de mi director de tesis Leonardo Chacon, por eso ejecuto la PINN varias veces con diferentes    parametros
* En esas ejecuciones tomo todos los datos posibles y los guardo en imagenes y archivos csv
* las versiones posteriores se volvera a mostrar los graficos de la PINN y solo ejecuatara un modelo como se tenia
* el commit esta en la fecha 06/04/2026

"""

"""

versión 28

* Esta es la EDP con cambio de variable (t) ha (τ), más que cambiar el caracter, el proposito es que en el nuevo residuo a resolver ya no explota en un tiempo finito, sino que se envio al infinito
* El resto del código sigue igual a como funciona la versión 26, con la diferencia que todo lo que era (t) se cambio a (τ) hasta en los comentarios, puede que en los comentaros se me pasara uno que otro
  t que se me olvidara quitar
* se dejo el puntero en la función de activación que viene desde la versión 27
* Tambien se cambiaron todos los (t) a (τ) en Graficos_PINN.py

"""


"""
CORRECCIÓN VERSIÓN 28

* Despues de una charla con mi direcotr de Tesis, el profesor Leonardo, el t0 definido en el articulo que era T_final en mi código siempre debio ser 0.3170 y (τ) debio ser independiente de ese T_final, nunca debio usarse como cota [0,T_final] ya que paa eso era el cambio de variable, elimianr esa restricción y dejar que crezca a 1,2,...,n
  En realidad T_final lo mal interprete tomandolo como la cota máxima de (τ) y dandole los valores que no le correspondia cuando nunca debio cambiar del 0.3170 de al explosión original, lo que hacia que el resultado de al EDP fuera SUPER ERRONEO
  Ahora la cota de (τ) esta dado por [0,maxvalτ] que SI ES EL QUE VARIA DE (1,.....,n) y T_final que ahora se llama T_blow_up queda fijo como 0.3170, respetando asi el cambio de variable y la estrucutra de este.
* Se crearon las variables minvalx,maxvalx,minvalτ,maxvalτ para que quede más comodo colocar los limites y cambiarlos en un solo lugar en vez de hacerlo linea por linea como se hacia antes
* Se creo una opción para decidir si activar el muestreo adaptativo o no desde la consola, esto es para no tener que hacerlo desde el código y sea más comodo


"""

"""
Una Ultimo ajuste
*En la versión de correción aun quedo algo mal, estaba mal definido el muestreo adaptativo que ya quedo corregido y comentado en al linea correspondiente del código
*se elimino alpha_pde y betha_pde (creo que asi estaban definidos), ya no estaban aportando nada a la nueva fisica
"""


"""
un juste más :/
*Se que sonara repetitivo, pero se encontro un error en los mapas de calor que cuando se extendia tao, , se modificaba el (-1,1) en x tambien, porque estaba usando el maxtau donde iba el tensor de x ._. el cual ya fue corregido
*la PINN crece un poco más cuando no esta amarrada a los pesos, aun asi los dejare en el código
*El muestreo adaptativo hasta el momento solo sirve para la PINN antes del cambio de variable, el problema original, para este no esta funcionando bien
*se cambio tf.square(x) por tf.pow(x,2), es lo mismo, pero prefiero este
"""