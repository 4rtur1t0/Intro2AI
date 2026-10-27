"""
    El algoritmo k-means programado desde cero con numpy.

    Notación: X es la matriz de datos (N, d), una fila por punto; C es la matriz de
    centroides (k, d), una fila por grupo; las etiquetas son un vector (N,) de enteros
    entre 0 y k-1 que dice a qué grupo pertenece cada punto.
"""
import numpy as np


def inicializar(X, k, semilla=0):
    """
    Elige como centroides iniciales k puntos de X distintos, al azar.
    """
    rng = np.random.default_rng(semilla)
    indices = rng.choice(len(X), size=k, replace=False)
    return X[indices].copy()


def asignar(X, C):
    """
    PASO 1 · ASIGNACIÓN. Devuelve, para cada punto de X, el índice del centroide de C
    más cercano (distancia euclídea). Resultado: vector (N,) de enteros.

    """
    ###########################################################
    # FORMA LENTA EN PYTHON PERO CONCEPTUALMENTE SIMPLE PARA EL ESTUDIANTE
    # Se anota aquí la forma más lenta de asignar (pero más fácil de entender).
    # distancias = np.zeros((len(X), len(C)))
    # asignaciones = np.zeros((len(X)))
    # cálculo de la distancia de cada punto al centroide j
    # for i in range(len(X)):
    #     for j in range(len(C)):
    #         distancias[i, j] = np.linalg.norm(X[i] - C[j])
        # cálculo de la distancia
    # for i in range(len(X)):
    #     asignaciones[i] = np.argmin(distancias[i, :])
    # return asignaciones
    ############################################################
    # FORMA RECOMENDADA!
    ################################################################################################
    # Numpy encapsula muchas operaciones de forma rápida (internamente se realizan en C/C++)
    # se hacen de forma más rápida, pues se evita la interpretación de tantas líneas en python.
    # A) (X - C[j]) calcula la diferencia de todos los puntos X al cluster j. Esta es una sintaxis curiosa de numpy,
    #    pues a una matriz de puntos X le resta un vector. Lo hace siempre en la dimension que coincide.
    # B) **2 calcula el cuadrado en las coordenadas X, Y y Z
    # C) np.sum(X - C[j]) ** 2, axis=1), suma X^2 + Y^2 + Z^2 por columnas (axis=1)
    #
    # Calculadas todas las distancias, se halla el índice de la mínima distancia en cada fila, esto corresponde
    # con el centroide más próximo a cada punto.
    ################################################################################################
    distancias = np.zeros((len(X), len(C)))
    for j in range(len(C)):
        distancias[:, j] = np.sum((X - C[j]) ** 2, axis=1)
    return np.argmin(distancias, axis=1)


def actualizar(X, etiquetas, C):
    """
    PASO 2 · ACTUALIZACIÓN. Devuelve los nuevos centroides (k, d): el centroide j pasa
    a ser la MEDIA de los puntos con etiqueta j. Si un grupo se ha quedado vacío,
    conserva su centroide anterior C[j].
    """
    #############################################
    # otra solución alternativa
    # for i in range(C.shape[0]):
        # cálculo del centroide i
        # pci = X[etiquetas == i]
        # if len(pci) == 0:
        #     continue
        # C[i] = np.mean(pci, axis=0)
    # return C
    ############################################
    nuevos = C.copy()
    for j in range(len(C)): # o range(C.shape[0])
        miembros = X[etiquetas == j]
        if len(miembros) > 0:
            nuevos[j] = miembros.mean(axis=0)
    return nuevos


def inercia(X, etiquetas, C):
    """
    Suma de las distancias AL CUADRADO de cada punto a su centroide. Es lo que k-means
    intenta hacer pequeño: cuanto menor, más compactos son los grupos.
    """
    return float(np.sum((X - C[etiquetas]) ** 2))


def kmeans(X, k, n_iter=100, semilla=0, C_inicial=None):
    """
    Agrupa los puntos de X en k grupos. Repite asignar + actualizar hasta que los
    centroides dejan de moverse o se alcanzan n_iter iteraciones.

    Si se da C_inicial, se parte de esos centroides; si no, de k puntos al azar.
    Devuelve (etiquetas, C, inercia, iteraciones realizadas).
    """
    ########################################################
    #   sol. un poco más corta
    ########################################################
    # for i in range(1, n_iter + 1):
    #     C_old = C.copy()
    #     labels = asignar(X, C)
    #     actualizar(X, labels, C)
    #     if np.allclose(C, C_old):
    #         break
    # actualizar(X, labels, C)
    # return labels, C, inercia(X=X, etiquetas=labels, C=C), i
    ########################################################
    C = inicializar(X, k, semilla) if C_inicial is None else np.array(C_inicial, dtype=float)
    for it in range(1, n_iter + 1):
        etiquetas = asignar(X, C)
        C_nuevo = actualizar(X, etiquetas, C)
        if np.allclose(C_nuevo, C):
            C = C_nuevo
            break
        C = C_nuevo
    etiquetas = asignar(X, C)
    return etiquetas, C, inercia(X, etiquetas, C), it
