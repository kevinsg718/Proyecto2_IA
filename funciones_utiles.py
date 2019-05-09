# En esta clase estan contenidas las funciones necesarias, que no forman parte directamente de la red neuronal
import numpy as np


def sigmoide(z):
    return 1.0 / (1.0 + np.exp(-z))


def derivada_sigmoide(z):
    sig = sigmoide(z)
    return sig * (1 - sig)


# la funcion de costo es error quadratico medio
# c = (1/(2n)) * sum(y - a)^2
# c' = (1/n) * sum(y - a)

def gradiente_funcion_costo(y, a):
    return a - y
