# En esta clase se leen las imagenes
import numpy as np
import random
from skimage import color, io


# Se cargan las imagenes obtenidas desde google 6000 del total
#       Se cargaron desde los archivos .npy y luego se procesaron para poder utilizarlas como se desea.
#       Estas imagenes vienen en RGB y con el negro siendo 0,0,0 y blanco 255,255,255. Para modificarlo
#       Se dividio dentro de 255 y se obtuvo una muestra pequena para no agotar la ram. En otras palabras.
#       Se obtuvo solo un slice del total de elementos que contenia el array de Google
def cargar_imagenes():
    conjunto_cargado = np.load('Imagenes/circulo.npy')[:6000] / 255.0  # se cargan 6000 para luego dividir en grupo y queden blanco y negro
    circulo = []
    for x in conjunto_cargado:
        circulo.append(np.reshape(x, (784, 1)))#.reshape(array,(columna,fila)) 784 por que es 28*28
    array_comparacion = np.zeros((10, 1))  # se crea un array con ceros, y con el tipo de imagen en un indice
    array_comparacion[0] = 1.0# array_comparacion (1.0, 0, 0, 0, ....)
    circulo_r = []
    for entrada in range(6000):
        circulo_r.append(array_comparacion) # circulo_r[(1.0, 0, 0, 0, ....), (1.0, 0, 0, 0, ....), (1.0, 0, 0, 0, ....)] 

    conjunto_cargado = np.load('Imagenes/cuadrado.npy')[:6000] / 255.0  # se cargan 6000 para luego dividir en grupo
    cuadrado = []
    for x in conjunto_cargado:
        cuadrado.append(np.reshape(x, (784, 1)))
    array_comparacion = np.zeros((10, 1))  # se crea un array con ceros, y con el tipo de imagen en un indice
    array_comparacion[1] = 1.0
    cuadrado_r = []
    for entrada in range(6000):
        cuadrado_r.append(array_comparacion)

    conjunto_cargado = np.load('Imagenes/triangulo.npy')[:6000] / 255.0  # se cargan 6000 para luego dividir en grupo
    triangulo = []
    for x in conjunto_cargado:
        triangulo.append(np.reshape(x, (784, 1)))
    array_comparacion = np.zeros((10, 1))  # se crea un array con ceros, y con el tipo de imagen en un indice
    array_comparacion[2] = 1.0
    triangle_result = []
    for entrada in range(6000):
        triangle_result.append(array_comparacion)

    conjunto_cargado = np.load('Imagenes/egg.npy')[:3000] / 255.0  # se cargan 6000 para luego dividir en grupo
    huevo = []
    for x in conjunto_cargado:
        huevo.append(np.reshape(x, (784, 1)))
    array_comparacion = np.zeros((10, 1))  # se crea un array con ceros, y con el tipo de imagen en un indice
    array_comparacion[3] = 1.0
    huevo_r = []
    for entrada in range(6000):
        huevo_r.append(array_comparacion)

    conjunto_cargado = np.load('Imagenes/arbol.npy')[:6000] / 255.0  # se cargan 6000 para luego dividir en grupo
    arbol = []
    for x in conjunto_cargado:
        arbol.append(np.reshape(x, (784, 1)))
    array_comparacion = np.zeros((10, 1))  # se crea un array con ceros, y con el tipo de imagen en un indice
    array_comparacion[4] = 1.0
    arbol_r = []
    for entrada in range(6000):
        arbol_r.append(array_comparacion)

    conjunto_cargado = np.load('Imagenes/casa.npy')[:6000] / 255.0  # se cargan 6000 para luego dividir en grupo
    casa = []
    for x in conjunto_cargado:
        casa.append(np.reshape(x, (784, 1)))
    array_comparacion = np.zeros((10, 1))  # se crea un array con ceros, y con el tipo de imagen en un indice
    array_comparacion[5] = 1.0
    casa_r = []
    for entrada in range(6000):
        casa_r.append(array_comparacion)

    conjunto_cargado = np.load('Imagenes/cara_feliz.npy')[:6000] / 255.0  # se cargan 6000 para luego dividir en grupo
    feliz = []
    for x in conjunto_cargado:
        feliz.append(np.reshape(x, (784, 1)))
    array_comparacion = np.zeros((10, 1))  # se crea un array con ceros, y con el tipo de imagen en un indice
    array_comparacion[6] = 1.0
    feliz_r = []
    for entrada in range(6000):
        feliz_r.append(array_comparacion)

    conjunto_cargado = np.load('Imagenes/question.npy')[:3000] / 255.0  # se cargan 6000 para luego dividir en grupo
    triste = []
    for x in conjunto_cargado:
        triste.append(np.reshape(x, (784, 1)))
    array_comparacion = np.zeros((10, 1))  # se crea un array con ceros, y con el tipo de imagen en un indice
    array_comparacion[7] = 1.0
    triste_r = []
    for entrada in range(6000):
        triste_r.append(array_comparacion)

    conjunto_cargado = np.load('Imagenes/sadface.npy')[:3000] / 255.0  # se cargan 6000 para luego dividir en grupo
    pregunta = []
    for x in conjunto_cargado:
        pregunta.append(np.reshape(x, (784, 1)))
    array_comparacion = np.zeros((10, 1))  # se crea un array con ceros, y con el tipo de imagen en un indice
    array_comparacion[8] = 1.0
    pregunta_r = []
    for entrada in range(6000):
        pregunta_r.append(array_comparacion)

    data_mickey = np.load('Imagenes/mouse.npy')[:2400] / 255.0  # se cargan 6000 para luego dividir en grupo
    sombra_mickey = []
    for x in data_mickey:
        sombra_mickey.append(np.reshape(x, (784, 1)))
    array_comparacion = np.zeros((10, 1))  # se crea un array con ceros, y con el tipo de imagen en un indice
    array_comparacion[9] = 1.0
    sombra_mickey_r = []
    for entrada in range(6000):
        sombra_mickey_r.append(array_comparacion)

    # ahora se unen los dos arrays creados por cada tipo de imagen
    imagenes = np.concatenate((circulo, cuadrado, triangulo, arbol, casa, feliz, sombra_mickey, pregunta, triste, huevo))
    #imagenes = np.concatenate((circulo, cuadrado, triangulo, huevo, arbol, casa, feliz, pregunta,  triste,sombra_mickey))
    respuestas = np.concatenate((circulo_r, cuadrado_r, triangle_result, arbol_r, casa_r, feliz_r, sombra_mickey_r, pregunta_r, triste_r, huevo_r))
    #respuestas = np.concatenate((circulo_r, cuadrado_r, triangle_result, huevo_r, arbol_r, casa_r, feliz_r, pregunta_r,  triste_r,sombra_mickey_r))
    imagenes_final = zip(imagenes, respuestas)  # Esto une las respuestas con las imagenes
    imagenes_final = list(imagenes_final)  # Esto crea una lista del objeto anterior

    # ahora se deben crear los subconjuntos de las imagenes train, cross, test
    random.shuffle(imagenes_final)
    train = imagenes_final[:int(0.80 * len(imagenes))]  # 80% del total de imagenes
    cross = imagenes_final[int(0.80 * len(imagenes)): int(0.80 * len(imagenes)) + int(0.10 * len(imagenes))]  # 10 que se usa para realizar pruebas
    test = imagenes_final[int(0.80 * len(imagenes)) + int(0.10 * len(imagenes)):]  # 10% de train
    return train, cross, test


def cargar_paint(im):
    p = 1 - color.rgb2gray(io.imread("Nuevo_Dibujo/" + im + ".bmp"))
    p = np.array(p).reshape(784, 1)#ajusto para que sea solo una fila
    p[:] = p[:] > 0#me devuelve lo que sea mayor a 0, en forma de array
    return p






