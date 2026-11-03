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
    for c in range(num_clases):
        X_c = X_train[y_train == c]             # solo los ejemplos de la clase c
        medias.append(np.mean(X_c, axis=0))
        desviaciones.append(np.std(X_c, axis=0))
        priors.append(len(X_c) / len(y_train))
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
    return (-0.5 * np.log(2 * np.pi * desviacion ** 2)
            - (x - media) ** 2 / (2 * desviacion ** 2))


def calcular_puntuaciones(x, medias, desviaciones, priors, num_clases):
    """
    La puntuación de cada clase c para el ejemplo x, un vector con sus n atributos:

        log P(c) + log P(x_1|c) + log P(x_2|c) + ... + log P(x_n|c)

    Es el logaritmo del numerador de la regla de Bayes. No hace falta dividir por P(x): es
    la misma para todas las clases y no cambia cuál es la mayor.
    :return: un diccionario {clase: puntuación}.
    """
    puntuaciones = {}
    for c in range(num_clases):
        p = np.log(priors[c])
        for i in range(len(x)):                 # atributo a atributo
            p += log_densidad_gaussiana(x[i], medias[c][i], desviaciones[c][i])
        puntuaciones[c] = p
    return puntuaciones


def predecir_naive_bayes(x, medias, desviaciones, priors, num_clases):
    """
    La clase del ejemplo x: la de mayor puntuación (la regla MAP).
    """
    puntuaciones = calcular_puntuaciones(x, medias, desviaciones, priors, num_clases)
    return max(puntuaciones, key=puntuaciones.get)
