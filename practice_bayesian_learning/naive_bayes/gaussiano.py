"""
    EJERCICIO 1 · NAÏVE BAYES GAUSSIANO.

    Con atributos continuos (los centímetros de una flor) no se puede contar: se supone que,
    dentro de cada clase c, cada atributo x_i sigue una campana de Gauss, con su media y su
    desviación típica, y P(x_i|c) es la densidad de esa campana:

                                   1                    (x_i - media)^2
        P(x_i | c) = --------------------------- exp(- ----------------- )
                      raiz(2 * pi) * desviacion           2 * desviacion^2

    Entrenar es calcular, para cada clase, P(c) y la media y la desviación típica de cada
    atributo, solo con los ejemplos de esa clase. Para clasificar un ejemplo x se calcula
    la PUNTUACIÓN de cada clase,

        puntuación(c) = log P(c) + log P(x_1|c) + log P(x_2|c) + ... + log P(x_n|c)

    y se elige la clase de mayor puntuación (la regla MAP). Se suman logaritmos en lugar de
    multiplicar probabilidades: el logaritmo no cambia cuál es la clase mayor, y así el
    resultado no se redondea a 0.

    Las clases se numeran: 0, 1, ..., num_clases - 1.
"""
import numpy as np


def entrenar_naive_bayes(X_train, y_train, num_clases):
    """
    Para cada clase c, calcula:
        - el vector con la media de cada atributo,
        - el vector con la desviación típica de cada atributo,
        - la probabilidad a priori P(c) = (n.º de ejemplos de c) / (n.º de ejemplos).
    :param X_train: array (n_ejemplos, n_atributos) con los atributos continuos.
    :param y_train: array (n_ejemplos,) con la clase de cada ejemplo: 0, 1, 2...
    :return: medias, desviaciones y priors, tres listas con un elemento por clase.
    """
    medias, desviaciones, priors = [], [], []
    ############################################################################
    # EJERCICIO 1 a): SE DEBE COMPLETAR EL CÓDIGO POR EL ESTUDIANTE.
    # Para cada clase c (0, 1, ..., num_clases - 1), y solo con los ejemplos de esa
    # clase, añada a cada lista su elemento: a medias, el vector con la media de cada
    # atributo; a desviaciones, el de las desviaciones típicas, y a priors, P(c), la
    # proporción de ejemplos de la clase c.
    # AYUDA: X_train[y_train == c] selecciona las filas de X_train de la clase c.
    #        np.mean(A, axis=0) y np.std(A, axis=0) calculan la media y la desviación
    #        típica de cada columna de A.
    ############################################################################
    raise NotImplementedError('EJERCICIO 1 a) sin completar: entrenar_naive_bayes()')
    return medias, desviaciones, priors


def log_densidad_gaussiana(x, media, desviacion):
    """
    El logaritmo de la densidad de una campana de Gauss con esa media y esa desviación
    típica, en el número x (es lo que calcula norm.logpdf(x, media, desviacion), de scipy):

                                                         (x - media)^2
        log P(x) = -0.5 * log(2 * pi * desviacion^2)  -  ----------------
                                                         2 * desviacion^2

    Se calcula directamente el logaritmo, y no la densidad y después su logaritmo: lejos de
    la media, la densidad es tan pequeña que el ordenador la redondea a 0.
    """
    ############################################################################
    # EJERCICIO 1 b): SE DEBE COMPLETAR EL CÓDIGO POR EL ESTUDIANTE.
    # Devuelva log P(x), con la fórmula de arriba.
    # AYUDA: np.log es el logaritmo neperiano; np.pi, el número pi, y a ** 2, el
    #        cuadrado de a.
    ############################################################################
    raise NotImplementedError('EJERCICIO 1 b) sin completar: log_densidad_gaussiana()')


def calcular_puntuaciones(x, medias, desviaciones, priors, num_clases):
    """
    La puntuación de cada clase c para el ejemplo x, un vector con sus n atributos:

        log P(c) + log P(x_1|c) + log P(x_2|c) + ... + log P(x_n|c)

    Es el logaritmo del numerador de la regla de Bayes. No hace falta dividir por P(x): es
    la misma para todas las clases y no cambia cuál es la mayor.
    :return: un diccionario {clase: puntuación}.
    """
    puntuaciones = {}
    ############################################################################
    # EJERCICIO 1 b): SE DEBE COMPLETAR EL CÓDIGO POR EL ESTUDIANTE.
    # Para cada clase c, su puntuación empieza en log P(c) y suma, atributo a
    # atributo, log P(x_i|c): la que calcula log_densidad_gaussiana para el valor
    # x[i], con la media y la desviación típica del atributo i en la clase c. Guarde
    # el resultado en puntuaciones[c].
    # AYUDA: medias[c] es el vector de medias de la clase c, así que medias[c][i] es
    #        la media del atributo i en esa clase (igual con desviaciones).
    ############################################################################
    raise NotImplementedError('EJERCICIO 1 b) sin completar: calcular_puntuaciones()')
    return puntuaciones


def predecir_naive_bayes(x, medias, desviaciones, priors, num_clases):
    """
    La clase del ejemplo x: la de mayor puntuación (la regla MAP).
    """
    puntuaciones = calcular_puntuaciones(x, medias, desviaciones, priors, num_clases)
    return max(puntuaciones, key=puntuaciones.get)
