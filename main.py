# Desde esta clase se va a llamar a las funciones de lectura y red neuronal
import lector
import redneuronal

if __name__ == '__main__':
    # - tamano de imagen 28x28. se escogio asi por votacion de las dos secciones. En teoria se encontro recomendable
    #   utilizarlo asi
    # - la salida seran 10 neuronas en el donde cada una representa el orden establecido por el profesor en la pagina
    #   de instrucciones. Circulo, cuadrado, triangulo, huevo, arbol, casa, cara feliz, cara triste, signo de interro-
    #   gacion, micky
    # - Arquitectura:
    #   - Capas ocultas, a definir.
    #   - Lambda: se utiliza para que el gradiente de descenso estocastico avance mas rapido o lento.
    #   - Funcion de activacion: sigmoide. Fue lo aprendido en clase, tiene una derivada sencilla
    # - Adquisición datos: Obtenidos grupalmente
    # - Preprosesamiento:
    #   - Para las imagenes de Google:
    #       Se cargaron desde los archivos .npy y luego se procesaron para poder utilizarlas como se desea.
    #       Estas imagenes vienen en RGB y con el negro siendo 0,0,0 y blanco 255,255,255. Para modificarlo
    #       Se dividio dentro de 255 y se obtuvo una muestra pequena para no agotar la ram. En otras palabras.
    #       Se obtuvo solo un slice del total de elementos que contenia el array de Google
    # - Cantidad de datos: (Pendiente)
    # - Problema de bias o variance? No se aun

    # CREAR LA RED NEURONAL
    parametros = [28*28, 30, 10]  # se ingresa un array con el numero de neuronas por capa
    red = redneuronal.RedNeuronal(parametros)

    # CARGAR LAS IMAGENES
    data = lector.cargar_imagenes()  # Data tiene 3 arrays de tuplas train, cross, test

    # ENTRENAR LA RED NEURONAL
    # se ingresa data, epocas (cuantas veces se va a 'aprender la data de train'), lote (tamano de subgrupo de
    # entrenamiento para el descenso de gradiente), lambda (velocidad de aprendizaje para el resultado S(z))
    red.entrenar(data, 25, 20, 2.0)

    # CREAR LOOP PARA ESPERAR IMAGENES DESDE PAINT
    while True:
        # esperar paint
        nueva = lector.cargar_paint()
        imagen, probabilidad = red.reconocer_imagen(nueva)

        # Clasificar resultado
        resultado = ""
        if imagen == 0:
            resultado = "circulo"

        elif imagen == 1:
            resultado = "cuadrado"

        elif imagen == 2:
            resultado = "triangulo"

        elif imagen == 3:
            resultado = "huevo"

        elif imagen == 4:
            resultado = "arbol"

        elif imagen == 5:
            resultado = "casa"

        elif imagen == 6:
            resultado = "cara feliz"

        elif imagen == 7:
            resultado = "cara triste"

        elif imagen == 8:
            resultado = "interrogacion"

        elif imagen == 9:
            resultado = "micky mouse"

        print("La imagen es: %s, con probabilidad de: %s" % (resultado, probabilidad))
