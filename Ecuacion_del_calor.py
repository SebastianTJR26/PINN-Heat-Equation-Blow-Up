import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Graficos_PINN import *
import numpy as np
import pandas as pd
import time
import os
import math
import random


#Toda operación que lleve el tf, esta ejecutanode en GPU (porque ya CUDA se enlazo con mi grafica NVIDIA y da true)
#Toda operación con np, esta en CPU

#PUNTERO: Menjear arreglos es lo más cercano a trabajar con las direcciones de memoria en python, como necesito cambiar la función de activación
#Para poder ver como les va a los demas experimentos con otras funciones, se crea una arreglo que contiene solo un dato tipo string en la posición 0 con el nombre de una función de activación
#Y como es un arreglo, cuando cambie la fcn de activación para una arquitectura, se que la red lo implementara porque "activation" esta hacia la dirección del string de l fcn de activación
puntero_fcn_activacion = ['relu'] #'tanh'
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

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def Creador_de_tensores_GlobalUniforme_IntervalosRandom(minvalx,maxvalx,minvalτ,maxvalτ,T_blow_up,beta,Puntos_dom): 

    # I Cree los arreglos primero linealmente espaceados

    if puntero_condicion_adaptattiva[0]=='S':      
        #(2) CON PONDERACIÓN ADAPTATIVA
        Adap0 = np.linspace(minvalx, maxvalx, Puntos_dom)
        X = np.sign(Adap0) * (np.abs(Adap0)**beta)
        AdapT = np.linspace(0.0, 1.0, Puntos_dom)
        T =  maxvalτ * (1.0 - (1.0 - AdapT)**beta)
    else:
        #(1) SIN PONDERACIÓN ADAPTATIVA
        X = np.linspace(minvalx, maxvalx, Puntos_dom)
        T = np.linspace(minvalτ, maxvalτ, Puntos_dom)

    
    # II Ahora haga el producto cartesiano entre ellos, para que existan  cordenadas (xi,τi) que cubran todo el dominio, si se deja sin hacer producto cartesiano, 
    # Solo cubrira la diagonal
    x_cart, τ_cart = np.meshgrid(X, T)

    # III Vuelvalos de nuevo en arreglos unidimensionales, porque como se hizo producto cartesiano, quedaron en forma de matriz
    prev1 = x_cart.reshape(-1,1)
    prev2 = τ_cart.reshape(-1,1)

    # IV Vamos a meter un punto aleatorio en cada sub intervalo de (x) y (τ) la razón por la que se hizo linealmente espaceado al inicio es porque las DF necesitan que sean asi
    # y se les mete el punto aleatorio porque asi lo necesita la definición de la PINN, de tal modo que cumpla con las dos y funcione bien la PINN+DF

    contador=1
    while contador<=2:
        if contador==1:
            nump_previo = prev1 #Primero identifiquemos si vamos a colocar los puntos aletorios de (x) que es 1 o (τ) que esta en el else
        else:
            nump_previo = prev2    
        llenar = []             #Despues cree el arreglo previo que vamos a llenar

        for i in range(len(nump_previo)-1):
            n = np.random.uniform(nump_previo[i], nump_previo[i+1]) #Esta bien hecho esto, logre meter un punto aleatorio dentro de cada intervalo linealmente espaciado
                                                                    #Cree un número aleatorio entre (i) (i+1)
            if i==0:
                llenar.append(nump_previo[i])
                llenar.append(n)                                    #inserte primero (i), luego el número aleatorio generado (n), y finalmente inserte (i+1)
                llenar.append(nump_previo[i+1])
        
            else:
                llenar.append(n)                                    #Solo necesitaba comenzar en (i) la primera vez, porque si vuelvo a empezar hay, se empieza a suscribir los números, ahora solo inserte (n) e (i+1) hasta terminar
                llenar.append(nump_previo[i+1])
        if contador==1:
            x = tf.convert_to_tensor(llenar, dtype=tf.float64)      #una vez listo el arreglo con los puntos alteatorios en cada sub intervalo, vuelvalo un tensor
        else:
            τ = tf.convert_to_tensor(llenar, dtype=tf.float64)
        contador+=1                            
    # V Retorne los arreglo generados
    return x,τ

"""
Al inicio se tenia hecho primero crear los arreglos linealmente espaceados, luego meterles el número aleatorio en cada sub intervalo y hay hacer el producto cartesiano, osea I->IV->II->III.
La idea era buena y lógica, pero por algun motivo, el dominio quedaba totalmente (linealmente espaceado) sin ningún número aleatorio en el intervalo.
Ahora en los pasos I->II->III->IV si garantizo tener el número aleatorio en cada sub intervalo

si recivo 99 puntos uniformes, para (x) y (τ), despues de que les haga el producto cartesiano a cada uno quedaran con 99x99= 9801 cada uno MÁS lOS 9800 NÚMEROS ALEATORIOS QUE VAN EN CADA UNO EN LOS SUB INTERVALOS
TOTAL PARA CADA UNO (x) y (τ) = 19000 puntos

"""         
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#segundo paso ,La función de pérdida es el núcleo de las PINNs. Combina tres componentes:
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#(1) Pérdida de la PDE: Mide cuánto se desvía la solución de la red neuronal de la ecuación diferencial. ∥∂u/∂t-α*∂²u/∂x²∥²
#...........................................................................................................................................................................................................................................................................
#antes, las condiciones iniciales y la frontera estaban en stack, no concatenadas, ahora ya si estan concatenadas

#La solución de la Ecuación Diferencial dada por la PINN se denoratara por: ---(u_φ)---

def EDP_loss(model, x, τ,w_ti,T_blow_up):
    # tf.GradientTape es una herramienta de TensorFlow para calcular derivadas (gradientes) de operaciones con respecto a sus variables.
    # with asegura que todas las operaciones dentro de él sean "grabadas" por GradientTape, para que luego se puedan calcular sus gradientes.
    # El parámetro persistent=True permite que la cinta (tape) sea reutilizable para calcular múltiples gradientes. 
    # Sin esto, la cinta se borraría después de calcular el primer gradiente.
    # tf.reduce_mean promedia estos errores para obtener una pérdida escalar.
    # tape.watch() le dice a TensorFlow que vigile o registre las operaciones que dependen de variables. 
    # Esto es necesario cuando variable no es un tensor entrenable (es decir, no es un parámetro de la red neuronal, 
    # como los pesos o biases), pero aún así necesitas calcular derivadas con respecto a esa variable.

    
    
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    #Este es al algoritomo de las Diferencias Finitas para la PINN
    Δx = 0.01  #Defina los deltas acorde a lo que explico el profe Leonardo siempre Δx<Δτ
    Δτ = 0.02


    f_x_τpΔτ=tf.concat([x, τ+Δτ], axis=1)
    f_x_τlΔτ=tf.concat([x, τ-Δτ], axis=1)  #Cree los tensores que va se van a evaluar en el model(), es decir, la función de la red
    f_xpΔx_τ=tf.concat([x+Δx, τ], axis=1)
    f_xlΔx_τ=tf.concat([x-Δx, τ], axis=1)
    f=tf.concat([x, τ], axis=1)
    
    du_dτ = (model(f_x_τpΔτ)-model(f_x_τlΔτ))/(2*Δτ)                # u_τ = (u(x, τ+Δτ)-u(x, τ-Δτ))/(2Δτ)
    d2u_dx2 = (model(f_xpΔx_τ)-2*model(f)+model(f_xlΔx_τ))/(Δx**2)  # u_xx = (u(x+Δx, τ)-2u(x, τ)+u(x-Δx, τ))/Δx²
    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
  
                                                                    #model(f)=u_φ que tenia antes, solo que ahora me toca dejarlo asi 
    residuo_edp = du_dτ - T_blow_up*tf.math.exp(-τ)*( d2u_dx2 + tf.pow(model(f),2) )  #nuevo residuo del cambio de variable  u_φ(x,t)->u_φ(x,τ) 
    
    if puntero_condicion_adaptattiva[0]=='S':

        #RETURN 2 CON PONDERACIÓN ADAPTATIVA
        return tf.reduce_sum(w_ti * tf.reduce_mean((residuo_edp)**2)) / tf.reduce_sum(w_ti) #-> (Σ w_ti * ||R||²)/ Σw_ti ESTA BIEN,  SI CUMPLE BIEN LA DEFINICIÓN
        #return tf.reduce_sum(w_ti * tf.square(residuo_edp)) / tf.reduce_sum(w_ti) #-> (Σ w_ti * √R)/ Σw_ti ESTA MAL, NO ESTA SACANDO LA NORMA AL CUADRADO

    else:

        #RETURN 1 SIN PONDERACIÓN ADAPTATIVA
        return tf.reduce_mean((residuo_edp)**2) 
    

    
# tf.reduce_mean((residuo_edp)**2) = tf.square(residuo_edp)

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
gamma = 2
 
 
T_blow_up = 0.3170  #CORRECIÓN: T_final nunca debio cambiar el valor de 0.3170, ya que el t0 dela artitulo es el tiempo finito de exploción y por ende la ecuación cuando se hizoo el cambio de variable
                    #Debe de entender que viene de una explosión finitia y en algun lado debe estar porque ambas ecuaciones estan correlacionadas, entones aunque tao vaya a infinito, T_final debio ser siempre 0.3170, ahora llamado T_blow_up

#Aqui definimos la cantidad de puntos para la PINN
Puntos_dom = 71
Puntos_ic= 1000
Puntos_bn = 500
# x ∈ (-1,1), τ>0 los imtervalos según la EDP con cambio de variable
minvalx = -1.0
maxvalx = 1.0
minvalτ = 0.0
maxvalτ = 1.0

if puntero_condicion_adaptattiva[0]=='S':
    beta = 2
else:
    beta = 0 


#---------------------------------------------------------------------------------------------------------------------------------------------
#Se volvio a usar funciones para tomar el muestreo de puntos porque de una forma necesito variar la cantidad de puntos para los n experimentos
#Si lo dejo de la otra forma no los podre variar por experimento
#---------------------------------------------------------------------------------------------------------------------------------------------
def puntos_muestro_pde(Puntos_dom, minvalx, maxvalx, minvalτ, maxvalτ, T_blow_up, beta):
    # Para los datos de la PDE, x ∈ (-1,1) τ>0, como de todos modos toca definir un intervalo final en τ, 
    # Se dejara hasta el tiempo donde se supone que explota
    x, τ = Creador_de_tensores_GlobalUniforme_IntervalosRandom(minvalx, maxvalx, minvalτ, maxvalτ , T_blow_up, beta, Puntos_dom)
    return x,τ



def puntos_muestro_ic(Puntos_ic, minvalx, maxvalx):
    # Para los datos en la condición inicial, x_in esta igual de definido que en x, es decir x_in ∈ (-1,1)
    # para τ_in todos los puntos generados son cero, puesto que comienzo en u(x,0), es por eso τ=0
    x_in = tf.random.uniform((Puntos_ic, 1), minval=minvalx, maxval=maxvalx, dtype=tf.float64)
    τ_in = tf.zeros_like(x_in, dtype=tf.float64) # en u(x_in,0) genere la cantidad de puntos donde esta definido x_in
    u0_func = funcion_valor_inicial(x_in)
    
    return x_in, τ_in, u0_func


def puntos_muestro_bn(Puntos_bn, minvalx, maxvalx, minvalτ, maxvalτ):
    # Left, Right
    # Para los datos en la frontera
    τ_bn  = tf.random.uniform((Puntos_bn, 1), minval=minvalτ, maxval=maxvalτ, dtype=tf.float64)  
    # porque como cuando definifimos x_in arriba, el x de la condición inicial esta (-1,1), 
    # el τ_bn esta creciendo, es decir ∈ t>0 (leer el articulo para comprenderlo)
    x_bnL = tf.ones_like(τ_bn, dtype=tf.float64) * minvalx#*-1 
    # en u(-1,τ_bn) genere la cantidad de puntos de t_bn para la frontera izquierda 
    x_bnR = tf.ones_like(τ_bn, dtype=tf.float64) * maxvalx#*1  
    # en u(1,τ_bn) genere la cantidad de puntos de t_bn para la frontera derecha             
    u_bn_func = tf.zeros_like(τ_bn, dtype=tf.float64)  
    # u(-1,τ_bn)=u(1,τ_bn)=0 para la función en la frontera, genere al comienzo, 
    # es decir en τ=0, los puntos generados
    
    return x_bnL, x_bnR, τ_bn, u_bn_func

#Necesito aunque sea una vez ver como se esta ejecutando la distribucion de puntos, si estoy en la ponderada al blow up o no
x, τ  = puntos_muestro_pde(Puntos_dom, minvalx, maxvalx, minvalτ, maxvalτ, T_blow_up, beta)
x_in,  τ_in, u0_func  = puntos_muestro_ic(Puntos_ic, minvalx, maxvalx)
x_bnL, x_bnR, τ_bn, u_bn_func= puntos_muestro_bn(Puntos_bn, minvalx, maxvalx, minvalτ, maxvalτ)
x_line = tf.linspace(tf.constant(minvalx, tf.float64), tf.constant(maxvalx, tf.float64), Puntos_ic)
u0_line = funcion_valor_inicial(tf.reshape(x_line, (-1,1)))
distribucion_dominio_y_condicion_inicial_GRAF(x,τ,minvalx,maxvalx,minvalτ,maxvalτ,x_in,τ_in,u0_func,x_line,u0_line,x_bnL,x_bnR,τ_bn,T_blow_up)

epsilon = 1e-3 #para cuando T_blow_up=τ, y no divida por 0, un valor demasiado minusculo para que no altere lo demas
w_ti = 1.0 / tf.pow(T_blow_up - τ + epsilon, gamma) #como esto no se esta derivando, no es necesario su verificación





#...........................................................................................................................................................................................................................................................................
#Entrenamiento ahora es una función porque necesito llamarla varias veces para los n experimentos, si lo dejo estatico como antes, solo correra una PINN
def Entrenamiento(model, x, τ, T_blow_up,x_in, τ_in, u0_func,x_bnL, x_bnR, τ_bn, u_bn_func,w_pde, w_ic, w_bc, w_ti, epochs):

    lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=[int(0.33*epochs), int(0.66*epochs)],values=[tf.constant(1e-2, tf.float64), tf.constant(1e-3, tf.float64), tf.constant(1e-4, tf.float64)]) 
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_sched) #(4)5e-4

    best_loss = 50.0 #guardamos el función de perdida más baja que dio, para saber cual fue el mejor entrenamiento, comenzamos en un valor muy alto para hacer comparación
    best_epoch = 0  #guardamos la epoca donde mejor fue entrenado el modelo
    

    for epoch in range(1, epochs + 1):

        with tf.GradientTape() as tape:
            loss_1 = EDP_loss(model, x, τ, w_ti, T_blow_up)
            loss_2 = Valores_Iniciales_loss(model, x_in, τ_in, u0_func)
            loss_3 = Frontera_loss(model, x_bnL, x_bnR, τ_bn, u_bn_func)

            PINN_loss = w_pde * loss_1 + w_ic * loss_2 + w_bc * loss_3

        gradients = tape.gradient(PINN_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        loss = PINN_loss.numpy()

        if loss < best_loss:
            best_loss = loss
            best_epoch = epoch
            xτ = tf.concat([x, τ], axis=1)
            u_pred = model(xτ)
            u_max = tf.reduce_max(tf.abs(u_pred)).numpy()
    

    return best_loss, best_epoch, u_max

#Esta función es la que recibe el arreglo de "resultados" despues de que cada experimento termine, y guarda todos los datos en el formato de una lista
#Esta es la forma correcta de crear una lista
def guardar_resultado(resultados,nombre,fcn_activacion,modelo,Puntos_dom,Puntos_ic,Puntos_bn,w_pde,w_ic,w_bc,epochs,maxvalτ,best_epoch,best_loss,u_max,tiempo_de_ejecucion):
    resultados.append({
        "fase": nombre,
        "fcn_activacion": fcn_activacion,
        "modelo": modelo,
        "puntos_dom": Puntos_dom,
        "puntos_ic": Puntos_ic,
        "puntos_bn": Puntos_bn,
        "w_pde": w_pde,
        "w_ic": w_ic,
        "w_bc": w_bc,
        "epochs": epochs,
        "τ_max": maxvalτ,
        "best_epoch": best_epoch,
        "best_loss": best_loss,
        "u_max": u_max,
        "tiempo_ejecución": str(tiempo_de_ejecucion) + " sg"
    })    

 


#...........................................................................................................................................................................................................................................................................
#EXPERIMENTOS
#¿Cómo funciona?

# (1)el primer ciclo estoy diciendo haciendo lo siguiente: quiero que me crees un arreglo que se llamara "resultados"
# (2)vamos a correr k experimentos, en este caso 15, si k esta en 16 es porque el segundo ciclo, el for termina en k-1 "asi funciona", ahora son 20 experimentos, porque voy a variar T_final, por tango k=21
# (3)vamos a iterar 3 veces y en esas 3 iteracciones vamos a hacer 15 experimentos de PINN, una donde se activen con tahn, otra con relu y otra con softplus
# en total, tendremos 45 experimentos
# (4) por cada grupo de 15 experimentos creamos un arreglo nuevo de resultados, esos resultados los vamos a guardar en resultados_n, y como ya quedaron guardados
# resultados_n no se vera afectado cuando pase al siguiente ciclo y el arreglo "resultados" se resetee de nuevo
# (5) OJO! Primero defino mi función de activación antes de iniciar cada experimento, porque si lo hago despues, es decir, corro "model" primero y luego cambio el puntero
# "model" estara corriendo la función de activación anterior, no la que acabe de cambiar
n=1
while(n<=1):


    
    resultados = []
    k=21
    print ("cantidad de experimentos a realizar",k-1)
    if n==1:
        resultados_1 = resultados #como estoy apuntando, lo que yo guarde en resultados ya queda guardado en resultados_1,resultados_2 o resultados_3 dependiendo del cilo, lo anterior no se elimina
        puntero_fcn_activacion[0] = 'tanh'
        print("comienzan los experimentos con tanh")
    """    
    elif n==2:
        resultados_2 = resultados
        puntero_fcn_activacion[0] = 'relu'
        print("comienzan los experimentos con relu")  #En realidad no tendria que descativar esto en comentarios, ya que si el ciclo solo esta hecho para correr una vez, no entrara en los otros elif
    elif n==3:                                        #es más para tener control de loq eu ya no estoy usando de momento
        resultados_3 = resultados
        puntero_fcn_activacion[0] = 'softplus'
        print("comienzan los experimentos con softplus")
    """    
    
#(a) Para el segundo ciclo defino los pesos de las 3 funciones de perdida "tambien podria experimentar con ellas variandolas, pero no lo vi relevante de momento"
#(b) definamos "best_loss" para cada experimento, esta en un valor muy grande porque la idea es encontrar uno más pequeño cada vez 
#(c) haga una variable para guardar la mejor época donde se registroe sa función de perdida
#(d) y comenze a correr el experimento "i". es muy intuitivo como esta definido cada experimento porque haz de cuenta que estoy definiendo para cada experimento "i"
#un entrenamiento como lo hacia antes, solo que aqui lo hago varias veces
#por alguna razon, no puedo obetenr los caracarteres de "model", asi que copiare esos caracteres en "modelo", solo es para ver que arquitectura era en la tabla
#una vez terminado el entrenamiento, guarde todo en el arreglo "resultados", y luego esos resultados se iran a "resultados_n"
#(e)cuando finalice los 15 experimentos, se reiniciara todo de nuevo para pasar a otra función de activación y capturar nuevos datos en "arreglo"
#los primeros 5 entrenamientos aumento las neuronas, los del 6-10 aumento los puntos en todo, y de 10-15 prolongo el entrenamiento con más épocas
   
    for i in range(1, k):
        w_pde = 0.3
        w_ic = 0.4
        w_bc = 0.3

#-----------------------VARIANDO LÍMITE TEMPORAL--------------------------------------------------------------

        if i==1:
            print("\n\n")
            print("VARIANDO LÍMITE TEMPORAL")
            print("\n\n")

            nombre = "experimento " + str(i)
            fcn_activacion = puntero_fcn_activacion[0]
            model = PINN([2,30, 30, 30, 30, 1])
            modelo = "[2,30, 30, 30, 30, 1]"

            maxvalτ = 1
            Puntos_dom = 72 # producto cruz da 5184 puntos más los 5183 en los sub intervalos da 10367
            Representacion_real_dom = 10367
            Puntos_ic= 1000
            Puntos_bn = 500
            epochs = 1000
            lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=[int(0.33*epochs), int(0.66*epochs)],values=[tf.constant(1e-2, tf.float64), tf.constant(1e-3, tf.float64), tf.constant(1e-4, tf.float64)])
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_sched)
            x, τ  = puntos_muestro_pde(Puntos_dom, minvalx, maxvalx, minvalτ, maxvalτ, T_blow_up, beta)
            x_in,  τ_in, u0_func  = puntos_muestro_ic(Puntos_ic, minvalx, maxvalx)
            x_bnL, x_bnR, τ_bn, u_bn_func= puntos_muestro_bn(Puntos_bn, minvalx, maxvalx, minvalτ, maxvalτ)

            print("experimento numero ",i," en ejecución")
            ini = time.perf_counter()
            best_loss, best_epoch, u_max= Entrenamiento(model, x, τ, T_blow_up,x_in, τ_in, u0_func,x_bnL, x_bnR, τ_bn, u_bn_func,w_pde, w_ic, w_bc, w_ti, epochs)
            end = time.perf_counter()
            print("finalizado\n")
            
            tiempo_de_ejecucion = end-ini

            guardar_resultado(resultados,nombre,fcn_activacion,modelo,Representacion_real_dom,Puntos_ic,Puntos_bn,w_pde,w_ic,w_bc,epochs,maxvalτ,best_epoch,best_loss,u_max,tiempo_de_ejecucion)

        elif i==2:
            nombre = "experimento " + str(i)
            fcn_activacion = puntero_fcn_activacion[0]
            model = PINN([2,30, 30, 30, 30, 1])
            modelo = "[2,30, 30, 30, 30, 1]"

            maxvalτ = 2
            Puntos_dom = 72
            Representacion_real_dom = 10367
            Puntos_ic= 1000
            Puntos_bn = 500
            epochs = 1000
            lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=[int(0.33*epochs), int(0.66*epochs)],values=[tf.constant(1e-2, tf.float64), tf.constant(1e-3, tf.float64), tf.constant(1e-4, tf.float64)])
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_sched)
            x, τ  = puntos_muestro_pde(Puntos_dom, minvalx, maxvalx, minvalτ, maxvalτ, T_blow_up, beta)
            x_in,  τ_in, u0_func  = puntos_muestro_ic(Puntos_ic, minvalx, maxvalx)
            x_bnL, x_bnR, τ_bn, u_bn_func= puntos_muestro_bn(Puntos_bn, minvalx, maxvalx, minvalτ, maxvalτ)

            print("experimento numero ",i," en ejecución")
            ini = time.perf_counter()
            best_loss, best_epoch, u_max= Entrenamiento(model, x, τ, T_blow_up,x_in, τ_in, u0_func,x_bnL, x_bnR, τ_bn, u_bn_func,w_pde, w_ic, w_bc, w_ti, epochs)
            end = time.perf_counter()
            print("finalizado\n")
            
            
            tiempo_de_ejecucion = end-ini

            guardar_resultado(resultados,nombre,fcn_activacion,modelo,Representacion_real_dom,Puntos_ic,Puntos_bn,w_pde,w_ic,w_bc,epochs,maxvalτ,best_epoch,best_loss,u_max,tiempo_de_ejecucion)

        elif i==3:
            nombre = "experimento " + str(i)
            fcn_activacion = puntero_fcn_activacion[0]
            model = PINN([2,30, 30, 30, 30, 1])
            modelo = "[2,30, 30, 30, 30, 1]"

            maxvalτ = 3
            Puntos_dom = 72
            Representacion_real_dom = 10367
            Puntos_ic= 1000
            Puntos_bn = 500
            epochs = 1000
            lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=[int(0.33*epochs), int(0.66*epochs)],values=[tf.constant(1e-2, tf.float64), tf.constant(1e-3, tf.float64), tf.constant(1e-4, tf.float64)])
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_sched)
            x, τ  = puntos_muestro_pde(Puntos_dom, minvalx, maxvalx, minvalτ, maxvalτ, T_blow_up, beta)
            x_in,  τ_in, u0_func  = puntos_muestro_ic(Puntos_ic, minvalx, maxvalx)
            x_bnL, x_bnR, τ_bn, u_bn_func= puntos_muestro_bn(Puntos_bn, minvalx, maxvalx, minvalτ, maxvalτ)

            print("experimento numero ",i," en ejecución")
            ini = time.perf_counter()
            best_loss, best_epoch, u_max= Entrenamiento(model, x, τ, T_blow_up,x_in, τ_in, u0_func,x_bnL, x_bnR, τ_bn, u_bn_func,w_pde, w_ic, w_bc, w_ti, epochs)
            end = time.perf_counter()
            print("finalizado\n")
            
            tiempo_de_ejecucion = end-ini

            guardar_resultado(resultados,nombre,fcn_activacion,modelo,Representacion_real_dom,Puntos_ic,Puntos_bn,w_pde,w_ic,w_bc,epochs,maxvalτ,best_epoch,best_loss,u_max,tiempo_de_ejecucion)

        elif i==4:
            nombre = "experimento " + str(i)
            fcn_activacion = puntero_fcn_activacion[0]
            model = PINN([2,30, 30, 30, 30, 1])
            modelo = "[2,30, 30, 30, 30, 1]"

            maxvalτ = 4
            Puntos_dom = 72
            Representacion_real_dom = 10367
            Puntos_ic= 1000
            Puntos_bn = 500
            epochs = 1000
            lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=[int(0.33*epochs), int(0.66*epochs)],values=[tf.constant(1e-2, tf.float64), tf.constant(1e-3, tf.float64), tf.constant(1e-4, tf.float64)])
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_sched)
            x, τ  = puntos_muestro_pde(Puntos_dom, minvalx, maxvalx, minvalτ, maxvalτ, T_blow_up, beta)
            x_in,  τ_in, u0_func  = puntos_muestro_ic(Puntos_ic, minvalx, maxvalx)
            x_bnL, x_bnR, τ_bn, u_bn_func= puntos_muestro_bn(Puntos_bn, minvalx, maxvalx, minvalτ, maxvalτ)

            print("experimento numero ",i," en ejecución")
            ini = time.perf_counter()
            best_loss, best_epoch, u_max= Entrenamiento(model, x, τ, T_blow_up,x_in, τ_in, u0_func,x_bnL, x_bnR, τ_bn, u_bn_func,w_pde, w_ic, w_bc, w_ti, epochs)
            end = time.perf_counter()
            print("finalizado\n")
            
            tiempo_de_ejecucion = end-ini

            guardar_resultado(resultados,nombre,fcn_activacion,modelo,Representacion_real_dom,Puntos_ic,Puntos_bn,w_pde,w_ic,w_bc,epochs,maxvalτ,best_epoch,best_loss,u_max,tiempo_de_ejecucion)

        elif i==5:
            nombre = "experimento " + str(i)
            fcn_activacion = puntero_fcn_activacion[0]
            model = PINN([2,30, 30, 30, 30, 1])
            modelo = "[2,30, 30, 30, 30, 1]"

            maxvalτ = 5
            Puntos_dom = 72
            Representacion_real_dom = 10367
            Puntos_ic= 1000
            Puntos_bn = 500
            epochs = 1000
            lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=[int(0.33*epochs), int(0.66*epochs)],values=[tf.constant(1e-2, tf.float64), tf.constant(1e-3, tf.float64), tf.constant(1e-4, tf.float64)])
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_sched)
            x, τ  = puntos_muestro_pde(Puntos_dom, minvalx, maxvalx, minvalτ, maxvalτ, T_blow_up, beta)
            x_in,  τ_in, u0_func  = puntos_muestro_ic(Puntos_ic, minvalx, maxvalx)
            x_bnL, x_bnR, τ_bn, u_bn_func= puntos_muestro_bn(Puntos_bn, minvalx, maxvalx, minvalτ, maxvalτ)

            print("experimento numero ",i," en ejecución")
            ini = time.perf_counter()
            best_loss, best_epoch, u_max= Entrenamiento(model, x, τ, T_blow_up,x_in, τ_in, u0_func,x_bnL, x_bnR, τ_bn, u_bn_func,w_pde, w_ic, w_bc, w_ti, epochs)
            end = time.perf_counter()
            print("finalizado\n")
            
            tiempo_de_ejecucion = end-ini

            guardar_resultado(resultados,nombre,fcn_activacion,modelo,Representacion_real_dom,Puntos_ic,Puntos_bn,w_pde,w_ic,w_bc,epochs,maxvalτ,best_epoch,best_loss,u_max,tiempo_de_ejecucion)

#-----------------------AUMENTANDO PUNTOS--------------------------------------------------------------

        elif i==6:
            print("\n\n")
            print("AUMENTANDO PUNTOS")
            print("\n\n")

            nombre = "experimento " + str(i)
            fcn_activacion = puntero_fcn_activacion[0]
            model = PINN([2,30, 30, 30, 30, 1])
            modelo = "[2,30, 30, 30, 30, 1]"

            maxvalτ = 5
            Puntos_dom = 78 # producto cruz da 6084 puntos más los 6083 en los sub intervalos da 12167
            Representacion_real_dom = 12167
            Puntos_ic= 1200
            Puntos_bn = 520
            epochs = 1000
            lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=[int(0.33*epochs), int(0.66*epochs)],values=[tf.constant(1e-2, tf.float64), tf.constant(1e-3, tf.float64), tf.constant(1e-4, tf.float64)])
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_sched)
            x, τ  = puntos_muestro_pde(Puntos_dom, minvalx, maxvalx, minvalτ, maxvalτ, T_blow_up, beta)
            x_in,  τ_in, u0_func  = puntos_muestro_ic(Puntos_ic, minvalx, maxvalx)
            x_bnL, x_bnR, τ_bn, u_bn_func= puntos_muestro_bn(Puntos_bn, minvalx, maxvalx, minvalτ, maxvalτ)

            print("experimento numero ",i," en ejecución")
            ini = time.perf_counter()
            best_loss, best_epoch, u_max= Entrenamiento(model, x, τ, T_blow_up,x_in, τ_in, u0_func,x_bnL, x_bnR, τ_bn, u_bn_func,w_pde, w_ic, w_bc, w_ti, epochs)
            end = time.perf_counter()
            print("finalizado\n")
            
            tiempo_de_ejecucion = end-ini

            guardar_resultado(resultados,nombre,fcn_activacion,modelo,Representacion_real_dom,Puntos_ic,Puntos_bn,w_pde,w_ic,w_bc,epochs,maxvalτ,best_epoch,best_loss,u_max,tiempo_de_ejecucion)

        elif i==7:
            nombre = "experimento " + str(i)
            fcn_activacion = puntero_fcn_activacion[0]
            model = PINN([2,30, 30, 30, 30, 1])
            modelo = "[2,30, 30, 30, 30, 1]"

            maxvalτ = 5
            Puntos_dom = 84 # producto cruz da 7056 puntos más los 7055 en los sub intervalos da 14111
            Representacion_real_dom = 14111
            Puntos_ic= 1400
            Puntos_bn = 540
            epochs = 1000
            lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=[int(0.33*epochs), int(0.66*epochs)],values=[tf.constant(1e-2, tf.float64), tf.constant(1e-3, tf.float64), tf.constant(1e-4, tf.float64)])
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_sched)
            x, τ  = puntos_muestro_pde(Puntos_dom, minvalx, maxvalx, minvalτ, maxvalτ, T_blow_up, beta)
            x_in,  τ_in, u0_func  = puntos_muestro_ic(Puntos_ic, minvalx, maxvalx)
            x_bnL, x_bnR, τ_bn, u_bn_func= puntos_muestro_bn(Puntos_bn, minvalx, maxvalx, minvalτ, maxvalτ)

            print("experimento numero ",i," en ejecución")
            ini = time.perf_counter()
            best_loss, best_epoch, u_max= Entrenamiento(model, x, τ, T_blow_up,x_in, τ_in, u0_func,x_bnL, x_bnR, τ_bn, u_bn_func,w_pde, w_ic, w_bc, w_ti, epochs)
            end = time.perf_counter()
            print("finalizado\n")
            
            tiempo_de_ejecucion = end-ini

            guardar_resultado(resultados,nombre,fcn_activacion,modelo,Representacion_real_dom,Puntos_ic,Puntos_bn,w_pde,w_ic,w_bc,epochs,maxvalτ,best_epoch,best_loss,u_max,tiempo_de_ejecucion)

        elif i==8:
            nombre = "experimento " + str(i)
            fcn_activacion = puntero_fcn_activacion[0]
            model = PINN([2,30, 30, 30, 30, 1])
            modelo = "[2,30, 30, 30, 30, 1]"

            maxvalτ = 5
            Puntos_dom = 90 # producto cruz da 8100 puntos más los 8099 en los sub intervalos da 16199
            Representacion_real_dom = 16199
            Puntos_ic= 1600
            Puntos_bn = 560
            epochs = 1000
            lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=[int(0.33*epochs), int(0.66*epochs)],values=[tf.constant(1e-2, tf.float64), tf.constant(1e-3, tf.float64), tf.constant(1e-4, tf.float64)])
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_sched)
            x, τ  = puntos_muestro_pde(Puntos_dom, minvalx, maxvalx, minvalτ, maxvalτ, T_blow_up, beta)
            x_in,  τ_in, u0_func  = puntos_muestro_ic(Puntos_ic, minvalx, maxvalx)
            x_bnL, x_bnR, τ_bn, u_bn_func= puntos_muestro_bn(Puntos_bn, minvalx, maxvalx, minvalτ, maxvalτ)

            print("experimento numero ",i," en ejecución")
            ini = time.perf_counter()
            best_loss, best_epoch, u_max= Entrenamiento(model, x, τ, T_blow_up,x_in, τ_in, u0_func,x_bnL, x_bnR, τ_bn, u_bn_func,w_pde, w_ic, w_bc, w_ti, epochs)
            end = time.perf_counter()
            print("finalizado\n")
            
            tiempo_de_ejecucion = end-ini

            guardar_resultado(resultados,nombre,fcn_activacion,modelo,Representacion_real_dom,Puntos_ic,Puntos_bn,w_pde,w_ic,w_bc,epochs,maxvalτ,best_epoch,best_loss,u_max,tiempo_de_ejecucion)

        elif i==9:
            nombre = "experimento " + str(i)
            fcn_activacion = puntero_fcn_activacion[0]
            model = PINN([2,30, 30, 30, 30, 1])
            modelo = "[2,30, 30, 30, 30, 1]"

            maxvalτ = 5
            Puntos_dom = 95 # producto cruz da 9025 puntos más los 9024 en los sub intervalos da 18049
            Representacion_real_dom = 18049
            Puntos_ic= 1800
            Puntos_bn = 580
            epochs = 1000
            lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=[int(0.33*epochs), int(0.66*epochs)],values=[tf.constant(1e-2, tf.float64), tf.constant(1e-3, tf.float64), tf.constant(1e-4, tf.float64)])
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_sched)
            x, τ  = puntos_muestro_pde(Puntos_dom, minvalx, maxvalx, minvalτ, maxvalτ, T_blow_up, beta)
            x_in,  τ_in, u0_func  = puntos_muestro_ic(Puntos_ic, minvalx, maxvalx)
            x_bnL, x_bnR, τ_bn, u_bn_func= puntos_muestro_bn(Puntos_bn, minvalx, maxvalx, minvalτ, maxvalτ)

            print("experimento numero ",i," en ejecución")
            ini = time.perf_counter()
            best_loss, best_epoch, u_max= Entrenamiento(model, x, τ, T_blow_up,x_in, τ_in, u0_func,x_bnL, x_bnR, τ_bn, u_bn_func,w_pde, w_ic, w_bc, w_ti, epochs)
            end = time.perf_counter()
            print("finalizado\n")
            
            tiempo_de_ejecucion = end-ini

            guardar_resultado(resultados,nombre,fcn_activacion,modelo,Representacion_real_dom,Puntos_ic,Puntos_bn,w_pde,w_ic,w_bc,epochs,maxvalτ,best_epoch,best_loss,u_max,tiempo_de_ejecucion)

        elif i==10:
            nombre = "experimento " + str(i)
            fcn_activacion = puntero_fcn_activacion[0]
            model = PINN([2,30, 30, 30, 30, 1])
            modelo = "[2,30, 30, 30, 30, 1]"

            maxvalτ = 5
            Puntos_dom = 101 # producto cruz da 10201 puntos más los 10200 en los sub intervalos da 20401
            Representacion_real_dom = 20401
            Puntos_ic= 2000
            Puntos_bn = 600
            epochs = 1000
            lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=[int(0.33*epochs), int(0.66*epochs)],values=[tf.constant(1e-2, tf.float64), tf.constant(1e-3, tf.float64), tf.constant(1e-4, tf.float64)])
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_sched)
            x, τ  = puntos_muestro_pde(Puntos_dom, minvalx, maxvalx, minvalτ, maxvalτ, T_blow_up, beta)
            x_in,  τ_in, u0_func  = puntos_muestro_ic(Puntos_ic, minvalx, maxvalx)
            x_bnL, x_bnR, τ_bn, u_bn_func= puntos_muestro_bn(Puntos_bn, minvalx, maxvalx, minvalτ, maxvalτ)

            print("experimento numero ",i," en ejecución")
            ini = time.perf_counter()
            best_loss, best_epoch, u_max= Entrenamiento(model, x, τ, T_blow_up,x_in, τ_in, u0_func,x_bnL, x_bnR, τ_bn, u_bn_func,w_pde, w_ic, w_bc, w_ti, epochs)
            end = time.perf_counter()
            print("finalizado\n")
            
            tiempo_de_ejecucion = end-ini

            guardar_resultado(resultados,nombre,fcn_activacion,modelo,Representacion_real_dom,Puntos_ic,Puntos_bn,w_pde,w_ic,w_bc,epochs,maxvalτ,best_epoch,best_loss,u_max,tiempo_de_ejecucion)

#-----------------------AUMENTANDO ARQUITECTURA--------------------------------------------------------------

        elif i==11:
            print("\n\n")
            print("AUMENTANDO ARQUITECTURA")
            print("\n\n")

            nombre = "experimento " + str(i)
            fcn_activacion = puntero_fcn_activacion[0]
            model = PINN([2,40, 40, 40, 40, 1])
            modelo = "[2,40, 40, 40, 40, 1]"

            maxvalτ = 5
            Puntos_dom = 101
            Representacion_real_dom = 20401
            Puntos_ic= 2000
            Puntos_bn = 600
            epochs = 1000
            lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=[int(0.33*epochs), int(0.66*epochs)],values=[tf.constant(1e-2, tf.float64), tf.constant(1e-3, tf.float64), tf.constant(1e-4, tf.float64)])
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_sched)
            x, τ  = puntos_muestro_pde(Puntos_dom, minvalx, maxvalx, minvalτ, maxvalτ, T_blow_up, beta)
            x_in,  τ_in, u0_func  = puntos_muestro_ic(Puntos_ic, minvalx, maxvalx)
            x_bnL, x_bnR, τ_bn, u_bn_func= puntos_muestro_bn(Puntos_bn, minvalx, maxvalx, minvalτ, maxvalτ)

            print("experimento numero ",i," en ejecución")
            ini = time.perf_counter()
            best_loss, best_epoch, u_max= Entrenamiento(model, x, τ, T_blow_up,x_in, τ_in, u0_func,x_bnL, x_bnR, τ_bn, u_bn_func,w_pde, w_ic, w_bc, w_ti, epochs)
            end = time.perf_counter()
            print("finalizado\n")
            
            tiempo_de_ejecucion = end-ini

            guardar_resultado(resultados,nombre,fcn_activacion,modelo,Representacion_real_dom,Puntos_ic,Puntos_bn,w_pde,w_ic,w_bc,epochs,maxvalτ,best_epoch,best_loss,u_max,tiempo_de_ejecucion)

        elif i==12:
            nombre = "experimento " + str(i)
            fcn_activacion = puntero_fcn_activacion[0]
            model = PINN([2,50, 50, 50, 50, 1])
            modelo = "[2,50, 50, 50, 50, 1]"

            maxvalτ = 5
            Puntos_dom = 101
            Representacion_real_dom = 20401
            Puntos_ic= 2000
            Puntos_bn = 600
            epochs = 1000
            lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=[int(0.33*epochs), int(0.66*epochs)],values=[tf.constant(1e-2, tf.float64), tf.constant(1e-3, tf.float64), tf.constant(1e-4, tf.float64)])
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_sched)
            x, τ  = puntos_muestro_pde(Puntos_dom, minvalx, maxvalx, minvalτ, maxvalτ, T_blow_up, beta)
            x_in,  τ_in, u0_func  = puntos_muestro_ic(Puntos_ic, minvalx, maxvalx)
            x_bnL, x_bnR, τ_bn, u_bn_func= puntos_muestro_bn(Puntos_bn, minvalx, maxvalx, minvalτ, maxvalτ)

            print("experimento numero ",i," en ejecución")
            ini = time.perf_counter()
            best_loss, best_epoch, u_max= Entrenamiento(model, x, τ, T_blow_up,x_in, τ_in, u0_func,x_bnL, x_bnR, τ_bn, u_bn_func,w_pde, w_ic, w_bc, w_ti, epochs)
            end = time.perf_counter()
            print("finalizado\n")
            
            tiempo_de_ejecucion = end-ini

            guardar_resultado(resultados,nombre,fcn_activacion,modelo,Representacion_real_dom,Puntos_ic,Puntos_bn,w_pde,w_ic,w_bc,epochs,maxvalτ,best_epoch,best_loss,u_max,tiempo_de_ejecucion)

        elif i==13:
            nombre = "experimento " + str(i)
            fcn_activacion = puntero_fcn_activacion[0]
            model = PINN([2,60, 60, 60, 60, 1])
            modelo = "[2,60, 60, 60, 60, 1]"

            maxvalτ = 5
            Puntos_dom = 101
            Representacion_real_dom = 20401
            Puntos_ic= 2000
            Puntos_bn = 600
            epochs = 1000
            lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=[int(0.33*epochs), int(0.66*epochs)],values=[tf.constant(1e-2, tf.float64), tf.constant(1e-3, tf.float64), tf.constant(1e-4, tf.float64)])
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_sched)
            x, τ  = puntos_muestro_pde(Puntos_dom, minvalx, maxvalx, minvalτ, maxvalτ, T_blow_up, beta)
            x_in,  τ_in, u0_func  = puntos_muestro_ic(Puntos_ic, minvalx, maxvalx)
            x_bnL, x_bnR, τ_bn, u_bn_func= puntos_muestro_bn(Puntos_bn, minvalx, maxvalx, minvalτ, maxvalτ)

            print("experimento numero ",i," en ejecución")
            ini = time.perf_counter()
            best_loss, best_epoch, u_max= Entrenamiento(model, x, τ, T_blow_up,x_in, τ_in, u0_func,x_bnL, x_bnR, τ_bn, u_bn_func,w_pde, w_ic, w_bc, w_ti, epochs)
            end = time.perf_counter()
            print("finalizado\n")
            
            tiempo_de_ejecucion = end-ini

            guardar_resultado(resultados,nombre,fcn_activacion,modelo,Representacion_real_dom,Puntos_ic,Puntos_bn,w_pde,w_ic,w_bc,epochs,maxvalτ,best_epoch,best_loss,u_max,tiempo_de_ejecucion)

        elif i==14:
            nombre = "experimento " + str(i)
            fcn_activacion = puntero_fcn_activacion[0]
            model = PINN([2,70, 70, 70, 70, 1])
            modelo = "[2,70, 70, 70, 70, 1]"

            maxvalτ = 5
            Puntos_dom = 101
            Representacion_real_dom = 20401
            Puntos_ic= 2000
            Puntos_bn = 600
            epochs = 1000
            lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=[int(0.33*epochs), int(0.66*epochs)],values=[tf.constant(1e-2, tf.float64), tf.constant(1e-3, tf.float64), tf.constant(1e-4, tf.float64)])
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_sched)
            x, τ  = puntos_muestro_pde(Puntos_dom, minvalx, maxvalx, minvalτ, maxvalτ, T_blow_up, beta)
            x_in,  τ_in, u0_func  = puntos_muestro_ic(Puntos_ic, minvalx, maxvalx)
            x_bnL, x_bnR, τ_bn, u_bn_func= puntos_muestro_bn(Puntos_bn, minvalx, maxvalx, minvalτ, maxvalτ)

            print("experimento numero ",i," en ejecución")
            ini = time.perf_counter()
            best_loss, best_epoch, u_max= Entrenamiento(model, x, τ, T_blow_up,x_in, τ_in, u0_func,x_bnL, x_bnR, τ_bn, u_bn_func,w_pde, w_ic, w_bc, w_ti, epochs)
            end = time.perf_counter()
            print("finalizado\n")
            
            tiempo_de_ejecucion = end-ini

            guardar_resultado(resultados,nombre,fcn_activacion,modelo,Representacion_real_dom,Puntos_ic,Puntos_bn,w_pde,w_ic,w_bc,epochs,maxvalτ,best_epoch,best_loss,u_max,tiempo_de_ejecucion)

        elif i==15:
            nombre = "experimento " + str(i)
            fcn_activacion = puntero_fcn_activacion[0]
            model = PINN([2,80, 80, 80, 80, 1])
            modelo = "[2,80, 80, 80, 80, 1]"

            maxvalτ = 5
            Puntos_dom = 101
            Representacion_real_dom = 20401
            Puntos_ic= 2000
            Puntos_bn = 600
            epochs = 1000
            lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=[int(0.33*epochs), int(0.66*epochs)],values=[tf.constant(1e-2, tf.float64), tf.constant(1e-3, tf.float64), tf.constant(1e-4, tf.float64)])
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_sched)
            x, τ  = puntos_muestro_pde(Puntos_dom, minvalx, maxvalx, minvalτ, maxvalτ, T_blow_up, beta)
            x_in,  τ_in, u0_func  = puntos_muestro_ic(Puntos_ic, minvalx, maxvalx)
            x_bnL, x_bnR, τ_bn, u_bn_func= puntos_muestro_bn(Puntos_bn, minvalx, maxvalx, minvalτ, maxvalτ)

            print("experimento numero ",i," en ejecución")
            ini = time.perf_counter()
            best_loss, best_epoch, u_max= Entrenamiento(model, x, τ, T_blow_up,x_in, τ_in, u0_func,x_bnL, x_bnR, τ_bn, u_bn_func,w_pde, w_ic, w_bc, w_ti, epochs)
            end = time.perf_counter()
            print("finalizado\n")
            
            tiempo_de_ejecucion = end-ini

            guardar_resultado(resultados,nombre,fcn_activacion,modelo,Representacion_real_dom,Puntos_ic,Puntos_bn,w_pde,w_ic,w_bc,epochs,maxvalτ,best_epoch,best_loss,u_max,tiempo_de_ejecucion)

#-----------------------AUMENTANDO ÉPOCAS--------------------------------------------------------------

        elif i==16:
            print("\n\n")
            print("AUMENTANDO ÉPOCAS")
            print("\n\n")
            nombre = "experimento " + str(i)
            fcn_activacion = puntero_fcn_activacion[0]
            model = PINN([2,80, 80, 80, 80, 1])
            modelo = "[2,80, 80, 80, 80, 1]"

            maxvalτ = 5
            Puntos_dom = 101
            Representacion_real_dom = 20401
            Puntos_ic= 2000
            Puntos_bn = 600
            epochs = 2000
            lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=[int(0.33*epochs), int(0.66*epochs)],values=[tf.constant(1e-2, tf.float64), tf.constant(1e-3, tf.float64), tf.constant(1e-4, tf.float64)])
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_sched)
            x, τ  = puntos_muestro_pde(Puntos_dom, minvalx, maxvalx, minvalτ, maxvalτ, T_blow_up, beta)
            x_in,  τ_in, u0_func  = puntos_muestro_ic(Puntos_ic, minvalx, maxvalx)
            x_bnL, x_bnR, τ_bn, u_bn_func= puntos_muestro_bn(Puntos_bn, minvalx, maxvalx, minvalτ, maxvalτ)

            print("experimento numero ",i," en ejecución")
            ini = time.perf_counter()
            best_loss, best_epoch, u_max= Entrenamiento(model, x, τ, T_blow_up,x_in, τ_in, u0_func,x_bnL, x_bnR, τ_bn, u_bn_func,w_pde, w_ic, w_bc, w_ti, epochs)
            end = time.perf_counter()
            print("finalizado\n")
            
            tiempo_de_ejecucion = end-ini

            guardar_resultado(resultados,nombre,fcn_activacion,modelo,Representacion_real_dom,Puntos_ic,Puntos_bn,w_pde,w_ic,w_bc,epochs,maxvalτ,best_epoch,best_loss,u_max,tiempo_de_ejecucion)


        elif i==17:
            nombre = "experimento " + str(i)
            fcn_activacion = puntero_fcn_activacion[0]
            model = PINN([2,80, 80, 80, 80, 1])
            modelo = "[2,80, 80, 80, 80, 1]"

            maxvalτ = 5
            Puntos_dom = 101
            Representacion_real_dom = 20401
            Puntos_ic= 2000
            Puntos_bn = 600
            epochs = 3000
            lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=[int(0.33*epochs), int(0.66*epochs)],values=[tf.constant(1e-2, tf.float64), tf.constant(1e-3, tf.float64), tf.constant(1e-4, tf.float64)])
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_sched)
            x, τ  = puntos_muestro_pde(Puntos_dom, minvalx, maxvalx, minvalτ, maxvalτ, T_blow_up, beta)
            x_in,  τ_in, u0_func  = puntos_muestro_ic(Puntos_ic, minvalx, maxvalx)
            x_bnL, x_bnR, τ_bn, u_bn_func= puntos_muestro_bn(Puntos_bn, minvalx, maxvalx, minvalτ, maxvalτ)

            print("experimento numero ",i," en ejecución")
            ini = time.perf_counter()
            best_loss, best_epoch, u_max= Entrenamiento(model, x, τ, T_blow_up,x_in, τ_in, u0_func,x_bnL, x_bnR, τ_bn, u_bn_func,w_pde, w_ic, w_bc, w_ti, epochs)
            end = time.perf_counter()
            print("finalizado\n")
            
            tiempo_de_ejecucion = end-ini

            guardar_resultado(resultados,nombre,fcn_activacion,modelo,Representacion_real_dom,Puntos_ic,Puntos_bn,w_pde,w_ic,w_bc,epochs,maxvalτ,best_epoch,best_loss,u_max,tiempo_de_ejecucion)

        elif i==18:
            nombre = "experimento " + str(i)
            fcn_activacion = puntero_fcn_activacion[0]
            model = PINN([2,80, 80, 80, 80, 1])
            modelo = "[2,80, 80, 80, 80, 1]"

            maxvalτ = 5
            Puntos_dom = 101
            Representacion_real_dom = 20401
            Puntos_ic= 2000
            Puntos_bn = 600
            epochs = 4000
            lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=[int(0.33*epochs), int(0.66*epochs)],values=[tf.constant(1e-2, tf.float64), tf.constant(1e-3, tf.float64), tf.constant(1e-4, tf.float64)])
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_sched)
            x, τ  = puntos_muestro_pde(Puntos_dom, minvalx, maxvalx, minvalτ, maxvalτ, T_blow_up, beta)
            x_in,  τ_in, u0_func  = puntos_muestro_ic(Puntos_ic, minvalx, maxvalx)
            x_bnL, x_bnR, τ_bn, u_bn_func= puntos_muestro_bn(Puntos_bn, minvalx, maxvalx, minvalτ, maxvalτ)

            print("experimento numero ",i," en ejecución")
            ini = time.perf_counter()
            best_loss, best_epoch, u_max= Entrenamiento(model, x, τ, T_blow_up,x_in, τ_in, u0_func,x_bnL, x_bnR, τ_bn, u_bn_func,w_pde, w_ic, w_bc, w_ti, epochs)
            end = time.perf_counter()
            print("finalizado\n")
            
            tiempo_de_ejecucion = end-ini

            guardar_resultado(resultados,nombre,fcn_activacion,modelo,Representacion_real_dom,Puntos_ic,Puntos_bn,w_pde,w_ic,w_bc,epochs,maxvalτ,best_epoch,best_loss,u_max,tiempo_de_ejecucion)

        elif i==19:
            nombre = "experimento " + str(i)
            fcn_activacion = puntero_fcn_activacion[0]
            model = PINN([2,80, 80, 80, 80, 1])
            modelo = "[2,80, 80, 80, 80, 1]"

            maxvalτ = 5
            Puntos_dom = 101
            Representacion_real_dom = 20401
            Puntos_ic= 2000
            Puntos_bn = 600
            epochs = 5000
            lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=[int(0.33*epochs), int(0.66*epochs)],values=[tf.constant(1e-2, tf.float64), tf.constant(1e-3, tf.float64), tf.constant(1e-4, tf.float64)])
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_sched)
            x, τ  = puntos_muestro_pde(Puntos_dom, minvalx, maxvalx, minvalτ, maxvalτ, T_blow_up, beta)
            x_in,  τ_in, u0_func  = puntos_muestro_ic(Puntos_ic, minvalx, maxvalx)
            x_bnL, x_bnR, τ_bn, u_bn_func= puntos_muestro_bn(Puntos_bn, minvalx, maxvalx, minvalτ, maxvalτ)

            print("experimento numero ",i," en ejecución")
            ini = time.perf_counter()
            best_loss, best_epoch, u_max= Entrenamiento(model, x, τ, T_blow_up,x_in, τ_in, u0_func,x_bnL, x_bnR, τ_bn, u_bn_func,w_pde, w_ic, w_bc, w_ti, epochs)
            end = time.perf_counter()
            print("finalizado\n")
            
            tiempo_de_ejecucion = end-ini

            guardar_resultado(resultados,nombre,fcn_activacion,modelo,Representacion_real_dom,Puntos_ic,Puntos_bn,w_pde,w_ic,w_bc,epochs,maxvalτ,best_epoch,best_loss,u_max,tiempo_de_ejecucion)

        elif i==20:
            nombre = "experimento " + str(i)
            fcn_activacion = puntero_fcn_activacion[0]
            model = PINN([2,80, 80, 80, 80, 1])
            modelo = "[2,80, 80, 80, 80, 1]"

            maxvalτ = 5
            Puntos_dom = 101
            Representacion_real_dom = 20401
            Puntos_ic= 2000
            Puntos_bn = 600
            epochs = 6000
            lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=[int(0.33*epochs), int(0.66*epochs)],values=[tf.constant(1e-2, tf.float64), tf.constant(1e-3, tf.float64), tf.constant(1e-4, tf.float64)])
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_sched)
            x, τ  = puntos_muestro_pde(Puntos_dom, minvalx, maxvalx, minvalτ, maxvalτ, T_blow_up, beta)
            x_in,  τ_in, u0_func  = puntos_muestro_ic(Puntos_ic, minvalx, maxvalx)
            x_bnL, x_bnR, τ_bn, u_bn_func= puntos_muestro_bn(Puntos_bn, minvalx, maxvalx, minvalτ, maxvalτ)

            print("experimento numero ",i," en ejecución")
            ini = time.perf_counter()
            best_loss, best_epoch, u_max= Entrenamiento(model, x, τ, T_blow_up,x_in, τ_in, u0_func,x_bnL, x_bnR, τ_bn, u_bn_func,w_pde, w_ic, w_bc, w_ti, epochs)
            end = time.perf_counter()
            print("finalizado\n")
            
            tiempo_de_ejecucion = end-ini

            guardar_resultado(resultados,nombre,fcn_activacion,modelo,Representacion_real_dom,Puntos_ic,Puntos_bn,w_pde,w_ic,w_bc,epochs,maxvalτ,best_epoch,best_loss,u_max,tiempo_de_ejecucion)
    
    #aumente en 1 para pasar a guardar la siguiente tabla, y asi hasta salir del ciclo, el while se rompe en n>3
    #Ahora solo se rompe despues del primer ciclo porque no voy a capturar datos de la fcn de las otras dos funciones de activación                                                  
    n = n + 1

#convierta esos resultados de los 20 esperimentos para tahn, relu y softplus y dataframes de pandas
DataFrame_PINN= pd.DataFrame(resultados_1)
print(DataFrame_PINN)

if puntero_condicion_adaptattiva[0]=='S':
    csv = "Experimentos_CB_DF_Muestreo_adaptativo.csv"
    imagen = "Experimentos_CB_DF_Muestreo_adaptativo.png"
    tituloo = "Tabla de resultados para la PINN DF con el cambio de variable y muestreo adaptativo"

else:
    csv = "Experimentos_CB_DF_SIN_muestreo_adaptativo.csv"
    imagen = "Experimentos_CB_DF_SIN_muestreo_adaptativo.png"
    tituloo = "Tabla de resultados para la PINN DF con el cambio de variable sin muestreo adaptativo"    

DataFrame_PINN.to_csv(csv, index=False)






#esta función genera las omagenes de las tablas, no me intereza saber como funciona, solo se que lo hace
#---------------------------------------------------------------------------------------------
def df_a_png(DataFrame_PINN, nombre_archivo="tabla.png", titulo="DataFrame", ancho=32, alto=None):
    import matplotlib.pyplot as plt

    n_filas = len(DataFrame_PINN)

    if alto is None:
        alto = 0.8 * n_filas + 2   

    fig, ax = plt.subplots(figsize=(ancho, alto))
    ax.axis("off")

    tabla = ax.table(
        cellText=DataFrame_PINN.values,
        colLabels=DataFrame_PINN.columns,
        cellLoc="center",
        loc="center"
    )

    # Estilo ajustado
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(13)        
    tabla.scale(1.2, 2.0)        

    # Colores tipo Excel
    for (fila, col), celda in tabla.get_celld().items():
        if fila == 0:
            celda.set_text_props(weight='bold', color='white')
            celda.set_facecolor('#4F81BD')
        else:
            celda.set_facecolor('#F2F2F2')

    plt.title(titulo, fontsize=16, weight='bold')
    plt.tight_layout()

    plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
    print(f"Imagen guardada como: {nombre_archivo}")

    plt.show()

#y genere las imagenes por cada dataframe, en este caso guarde 3 por las funciones de activación tanh, relu y softplus
#FIN DE ESTE CÓDIGO (SUGERENCIA: PODRIA DOCUMENTARLO AL ESTILO DE LEONARDO VALENCIA CON END_WHILE)
df_a_png(DataFrame_PINN , imagen, tituloo)




  





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
versión 29 (nombre del commit : Experimentos con la PINN cambio de variable (TABLAS))

* Es igual que la versión 27, pero a diferencia que esta saca los resultados de varios experimentos de la PINN con el vambio de variable (x,τ)
* el commit esta en la fecha 08/04/2026

"""


"""
versión 30 o 1.2

*Esta debio ser el primer código cuando se arranco la PINN, porque este muestra como mi red soluciona la EDP con u_0 = sin(pi*x) y las condiciones de Dirichlet
*Ademas este problema es el que tiene la solución analítica para comparar si estaba bien o no
*se grafico la solución de la PINN vs la real y efectivamente, mi red cumple con la solución, aunque la analítica pues siempre sera la mejor porque es la solución real
*se tiene comentado todo lo que hizo, tanto las modificaciones,experimentos y nuevo código que se añado para esta versión
*por algún motivo, el código en la parte de pintar la distribución de puntos no me dejo, aparece un error de tamaños, pero la verdad no interesa tanto solucionar eso porque ya comprobando el vs con la solución analítica->
-> y viendo que el error es muy minimo y son casi identicas, no me preocupa graficar esa dsitribución entonces, porque ya se que esta bien con la comparación de la solución real 

"""

"""
versión 31

*Se añadio la solución tambien de las Diferencias finitas con las condiciones y el nucleo de Dirichlet, para comparar los 3 mapas de calor, esto por instrucción del profesor Leoanrdo Chacon
*se hicieron unos cambios minimos para que el archivo de (Diferencias_finitas.py) solucionara la ecuación del calor original, al igual se dejo comentado en ese código que se toco y modifico para poder llegar a ello
*tambien se hicieron unos cambios en (Graficos_PINN.py) para mostrar los 3 mapas, algo muy minusculo y con comentarios sencillos
*Se corrigieron algunos comentarios que estaban mal en (Graficos_PINN.py), como que el segundo subplot que coloque la solución real, nunca le cambie el comentario que decia (solución diferencias finitas, o algo asi) 

"""


"""
versión 32

*PINN+DF
*Esta es la versión de la PINN donde la EDP_loss() Ya no trabaja con la DA(tape.gradient()), sino con las definiciones de las DF, Es mucho más rapido!!!
*Ahora (x, τ) trabajan Lnealmente espaceadas como se tenia a los inicios de la PINN, porque las DF los necesitan, pero con Puntos aleatorios en cada intervalo porque las PINN los necesitan
*Todos estos cambios mencionados aqui estan documentos en EDP_loss() y Creador_de_tensores_GlobalUniforme_IntervalosRandom()
*Aun opción de acumular los puntos a T_final como se hacia antes, o no, eso depende de cuando se esten haciendo los ejercicios de las tablas 
"""


"""
versión 33

*Acumulaciones en x=0

"""

"""
versión 34 (Correción de la 32 y cuando todavia era DA)

*Un error que se tenia cuando aumentaba el limite de tao de 1,2,...,n era que el t0 debio ser siempre el tiempo de explosión finito de la ecuación original en (x,t)
*Claro que tenia mis sospechas desde que implemente el residuo, pero hasta que el al profe Leonardo lo confirmo me di cuenta que desde la implementación con DA los resultados no estan bien y esas tablas tampoco
*Ahora fue correjido (SOLO LA DE DF), la de diferencias automaticas se creara una versión alterna y luego hacer de nuevo las tablas
*se agreggo una forma más sencillas ara amrcar los intervalos y no tener que cambiarlos manualmente para cada variable
*Se agrego una opción para desde la terminal se quiera activar el muestreo adaptativo o no

"""


"""
versión 35

* Se hizo una correción a los mapas de calor que hacia extender incorrectamente el dominio en (x) cuando crecia (tao), ver (PP) arriba, ya esta bien
* Aunque se este concentrando puntos al final de (tao) y en x=0, con o sin los pesos w_ti no se esta obteniendo buen resultado
* Esta deberia ser la versión final de la PINN almenos que se peuda corregir lo del muestreo
* Se dejan el mapa de diferencias finitas (x_vis,τ_vis)=(100,500) apra ver la explosión, es la recomendación de Brian
* La razón por la que la abrra de calor del mapa y el u_max no coincidene s porque el tensor que se envia al model viene de los (x_vis,τ_vis), pero ajustar eso para que coincidan esta un poco complicado y tampoco creo que valga la pena hacerlo
"""


"""
versión 36

* Esta es la versión de experimentos de la PINN+DF, esto es para completar el ciclo
-Experimentos PINN clásica
-Experimentos PINN muestreo adaptativo y ponderación temporal
-Experimentos PINN cambio de variable
-Experimentos PINN+DF cambio de variable
"""