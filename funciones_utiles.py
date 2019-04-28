# En esta clase estan contenidas las funciones necesarias, que no forman parte directamente de la red neuronal
import numpy as np


def sigmoide(z):
    return 1.0 / (1.0 + np.exp(-z))
