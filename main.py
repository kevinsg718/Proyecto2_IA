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
    #   - Lambda: no definido
    #   - Funcion de activacion: sigmoide. Fue lo aprendido en clase, tiene una derivada sencilla
    # - Adquisición datos: Obtenidos grupalmente
    # - Preprosesamiento: (ausente)
    # - Cantidad de datos: (Pendiente)
    # - Problema de bias o variance? No se aun

    # CREAR LA RED NEURONAL
    parametros = [28*28, 16, 14, 10]
    red = redneuronal.RedNeuronal(parametros)

    # CARGAR LAS IMAGENES
    data = lector.cargar_imagenes()

    # ENTRENAR LA RED NEURONAL
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
