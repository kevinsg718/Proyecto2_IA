# Red neuronal

import numpy as np, funciones_utiles


class RedNeuronal(object):
    # configurar la red neuronal con sus parametros: Numero de capas y neuronas por capa ej: [4, 5, 6]
    # 4 neuronas de entrada, 5 en una capa oculta y 6 en la capa de salida
    def __init__(self, capas):
        self.no_capas = len(capas)
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
        pass

    def feedforward(self, x):
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, x) + b
            x = funciones_utiles.sigmoide(z)
        return x

    def backpropagation(self):
        pass

    def gradient_descend(self):
        pass

    def funcion_costo(self):
        pass

    def gradiente_funcion_costo(self):
        pass

    def reconocer_imagen(self, imagen):
        self.feedforward(imagen)
        return 1, 0.4
