"""
    EJERCICIO 3 · NAÏVE BAYES MULTINOMIAL.

    Es la variante que se usa con texto. Los atributos son los CONTEOS de cada palabra del
    vocabulario en un mensaje (su bolsa de palabras), y P(palabra|c) se estima contando,
    con el suavizado de Laplace:

                              N(palabra, c) + alpha
        P(palabra | c) = --------------------------------
                           N(c) + alpha * |vocabulario|

    N(palabra, c) es el número de veces que aparece la palabra en TODOS los mensajes de
    la clase c, y N(c), el número total de palabras de la clase c. Sin el suavizado
    (alpha = 0), una palabra que no ha salido nunca en una clase tendría probabilidad 0
    en ella.

    Para clasificar un mensaje x se calcula la PUNTUACIÓN de cada clase, con logaritmos,

        puntuación(c) = log P(c) + sum_i  x[i] * log P(palabra_i | c)

    con x[i] el número de veces que la palabra i aparece en el mensaje: si aparece 3 veces,
    su logaritmo se suma 3 veces, y si no aparece, ninguna. Se elige la clase de mayor
    puntuación.

    Las clases se numeran: 0 es ham (legítimo), y 1, spam.
"""
import numpy as np


def entrenar_naive_bayes_multinomial(X_train, y_train, num_clases, alpha=1.0):
    """
    Para cada clase c, calcula log P(c) y log P(palabra|c) de cada palabra del vocabulario.
    Entrenar sigue siendo CONTAR.
    :param X_train: matriz de conteos (n_mensajes, n_palabras), la de la bolsa de palabras.
    :param y_train: array (n_mensajes,) con la clase de cada mensaje: 0 o 1.
    :return: log_priors, un array (num_clases,), y log_prob_palabras, un array
             (num_clases, n_palabras): log_prob_palabras[c][i] es log P(palabra i|c).
    """
    num_docs, num_palabras = X_train.shape
    log_priors = np.zeros(num_clases)
    log_prob_palabras = np.zeros((num_clases, num_palabras))
    ############################################################################
    # EJERCICIO 3 a): SE DEBE COMPLETAR EL CÓDIGO POR EL ESTUDIANTE.
    # Para cada clase c, y solo con los mensajes de esa clase:
    #   - log_priors[c]: el logaritmo de la proporción de mensajes de la clase c;
    #   - log_prob_palabras[c]: log P(palabra|c) de todas las palabras, con la
    #     fórmula de arriba (num_palabras es el tamaño del vocabulario).
    # AYUDA: X_train[y_train == c] selecciona los mensajes de la clase c, y
    #        A.sum(axis=0) suma cada columna de A: da N(palabra, c) de cada palabra,
    #        sin ningún bucle. Las operaciones con ese vector (sumarle un número,
    #        dividirlo, np.log) se hacen palabra a palabra.
    ############################################################################
    raise NotImplementedError('EJERCICIO 3 a) sin completar: entrenar_naive_bayes_multinomial()')
    return log_priors, log_prob_palabras


def calcular_puntuaciones_multinomial(test_sample, log_priors, log_prob_palabras):
    """
    La puntuación de cada clase c para un mensaje ya convertido en números, test_sample
    (su fila de la matriz de conteos):

        log P(c) + sum_i  test_sample[i] * log P(palabra_i | c)

    :return: un array con la puntuación de cada clase.
    """
    num_clases = len(log_priors)
    num_palabras = len(test_sample)
    puntuacion = np.zeros(num_clases)
    ############################################################################
    # EJERCICIO 3 b): SE DEBE COMPLETAR EL CÓDIGO POR EL ESTUDIANTE.
    # Para cada clase c, puntuacion[c] empieza en log P(c) y suma, palabra a palabra
    # del vocabulario, las veces que la palabra aparece en el mensaje por su
    # log P(palabra|c).
    # AYUDA: test_sample[i] es el número de veces que aparece en el mensaje la
    #        palabra de índice i, y log_prob_palabras[c][i], su log P(palabra|c).
    ############################################################################
    raise NotImplementedError('EJERCICIO 3 b) sin completar: '
                              'calcular_puntuaciones_multinomial()')
    return puntuacion


def predecir_naive_bayes_multinomial(test_sample, log_priors, log_prob_palabras):
    """
    La clase del mensaje: la de mayor puntuación (np.argmax devuelve su índice).
    """
    puntuacion = calcular_puntuaciones_multinomial(test_sample, log_priors, log_prob_palabras)
    return int(np.argmax(puntuacion))


def palabras_mas_frecuentes(log_prob_palabras, palabras, n=10):
    """
    Para cada clase, las n palabras con mayor P(palabra|c): las más habituales de la
    clase... que suelen ser también las más habituales de la otra.
    """
    resultado = []
    for c in range(len(log_prob_palabras)):
        orden = np.argsort(log_prob_palabras[c])[::-1][:n]      # de mayor a menor
        resultado.append([palabras[i] for i in orden])
    return resultado


def palabras_mas_indicativas(log_prob_palabras, palabras, n=10):
    """
    Las palabras que más EMPUJAN la decisión hacia el spam o hacia ham, medidas por su
    peso, la diferencia

        log P(palabra|spam) - log P(palabra|ham)

    Una palabra puede ser frecuentísima en el spam y no servir para nada si es igual de
    frecuente en los mensajes legítimos (por ejemplo, "the").
    Devuelve (las n de mayor peso, que empujan hacia el spam, y las n de menor peso, que
    empujan hacia ham).
    """
    peso = log_prob_palabras[1] - log_prob_palabras[0]
    orden = np.argsort(peso)                        # de menor a mayor peso
    return ([palabras[i] for i in orden[::-1][:n]],
            [palabras[i] for i in orden[:n]])
