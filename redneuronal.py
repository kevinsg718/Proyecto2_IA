# Red neuronal

import funciones_utiles
import numpy as np
from random import *


class RedNeuronal(object):
    # configurar la red neuronal con sus parametros: Numero de capas y neuronas por capa ej: [4, 5, 6]
    # 4 neuronas de entrada, 5 en una capa oculta y 6 en la capa de salida
    def __init__(self, capas):
        self.capas = capas

        # iniciar los bias random
        self.biases = []
        for i in capas[1:]:  # se inicia desde capa 2 porque la primera no tiene bias
            self.biases.append(np.random.randn(i, 1))  # crea un array con el tamano de capa de aleatorios

        # iniciar los pesos random
        self.weights = []
        for neurona1, neurona2 in zip(capas[:-1], capas[1:]):  # entre la capa de enfrente y la siguiente
            self.weights.append(np.random.randn(neurona2, neurona1))  # inicia los pesos, la ultima capa no tiene

    def entrenar(self, data, epocas, lote, lambd):
        #data
        #10 capas
        #20 lotes por que se entrena por pedasos, o pasos
        #2.0 learinig rate o lambda, es la velocidad de aprendizaje (lo saque probando cuanto tuvo de error)

        # dividir las tuplas en los diferentes grupos
        train = data[0]
        cross = data[1]
        test = data[2]

        # Convertir el resultado de test en numeros
        resultado_test = [np.argmax(y[1]) for y in test]#devolver el que tenga mayor

        # el proceso de entrenamiento se realizara la varias veces
        for epoca in range(epocas):
            # se crean los lotes para realizar el descenso de gradiente
            shuffle(train)
            lotes =[]
            for k in range(0,len(train),lote):
                lotes.append(train[k:k + lote])#se suman los lotes o pedasos de entrenamiento

            # por cada lote o paso hago descenso gradiente
            # se ejecuta el descenso de gradiente
            for i in lotes:
                self.gradient_descend(i, lambd)#pedaso y velocida que avanza

            # ahora se quiere calcular como le fue en la presente epoca
            # para esto se utiliza el subconjunto de test
            test_results = []
            for (x, y) in test:
                test_results.append(np.argmax(self.feedforward(x)))

            aciertos = 0
            for (resultado, valor_esperado) in zip(test_results, resultado_test):
                if resultado == valor_esperado:
                    aciertos += 1

            porcentaje = float(aciertos) / len(test) * 100
            print("Se obtuvo %s%% porcentaje de aciertos en la epoca %s" % (porcentaje, epoca))

    def feedforward(self, neuronas):#neuronas son imagen
        for index in range(len(self.biases)):
            # Multiplicar entrada por pesos y sumarle el bias
            z = np.dot(self.weights[index], neuronas) + self.biases[index]#W1*X1 + bias

            # aplicar el resultado de sigmoide
            neuronas = funciones_utiles.sigmoide(z)#aplico funcion de activacion a sumatoria z

        return neuronas

    def backpropagation(self, entrada, resultado):#imagen y resultado
        # Crear las variables para gradiente de pesos y bias
        gradiente_bias = []
        for b in self.biases:
            gradiente_bias.append(np.zeros(b.shape))

        gradiente_weight = []
        for w in self.weights:
            gradiente_weight.append(np.zeros(w.shape))

        # feedforward
        thetas = [entrada]  # se guardan todos los pesos por capa
        zetas = []

        for i in range(len(self.biases)):
            z = np.dot(self.weights[i], entrada) + self.biases[i]#producto entre las matrices entre pesos y la imagen sumo el bias W1*X1 + b
            entrada = funciones_utiles.sigmoide(z)#se a sigmoide al sumatoria(z)
            zetas.append(z)
            thetas.append(entrada)

        # obtener el rendimiento de la ultima capa
        ultima_capa = thetas[-1]
        #aplico funicion de coste para obtener el error
        #gradiente_funcion_costo (ultima_capa - resultado)
        #derivada_sigmoide(sigmoide(ultima entrada))

        L = funciones_utiles.gradiente_funcion_costo(resultado, ultima_capa) * funciones_utiles.derivada_sigmoide(
            zetas[-1])

        # backpropagation
        # iniciar las matrices de deltas con la ultima capa de pesos y biases
        gradiente_bias[-1] = L
        gradiente_weight[-1] = np.dot(L, thetas[-2].transpose())  # Propagar a los ultimos pesos
        
        # obtener las derivadas del resto de capas para continuar el calculo de coste
        for capa in range(2, len(self.capas)):
            L = np.dot(self.weights[-capa + 1].transpose(), L) * funciones_utiles.derivada_sigmoide(zetas[-capa])
            gradiente_bias[-capa] = L
            gradiente_weight[-capa] = np.dot(L, thetas[-capa - 1].transpose())

        return gradiente_bias, gradiente_weight

    def gradient_descend(self, data, lambd):#pedaso que voy a entrenar, velocidad de entrenamiento

        # Crear las variables donde se guardaran los pesos y bias actualizados
        gradiente_bias = []
        for b in self.biases:
            gradiente_bias.append(np.zeros(b.shape))#(n,m)

        gradiente_weight = []
        for w in self.weights:
            gradiente_weight.append(np.zeros(w.shape))

        # por cada tupla imagen-resultado
        for imagen, resultado_esperado in data:#es una tupla 
            # realizar feedforward/backpropagation con la tupla
            ajuste_bias, ajuste_weight = self.backpropagation(imagen, resultado_esperado)

            # despues de obtener el ajuste lo sumo al bias 
            for i in range(len(ajuste_bias)):
                gradiente_bias[i] = (gradiente_bias[i] + ajuste_bias[i])

            # obtener el gradiente de los weights lo sumo al peso
            for i in range(len(ajuste_weight)):
                gradiente_weight[i] = (gradiente_weight[i] + ajuste_weight[i])

        #Aqui hago el ajuste de las variables peso y bias
        # actualizar los pesos y bias de la red sumandole el delta calculado
        ajuste_lambda = (lambd / len(data))
        for i in range(len(gradiente_weight)):
            self.weights[i] = (self.weights[i] - ajuste_lambda * gradiente_weight[i])

        for i in range(len(gradiente_bias)):
            self.biases[i] = (self.biases[i] - ajuste_lambda * gradiente_bias[i])

    def reconocer_imagen(self, imagen):
        resultado = self.feedforward(imagen)#[0,1,0,1,1,0,...]
        imagen = np.argmax(resultado) #devuelve el indice de la probabilidad mas alta
        probabilidad = resultado[imagen]# devuelve el valor que esta en ese indice
        return imagen, probabilidad
