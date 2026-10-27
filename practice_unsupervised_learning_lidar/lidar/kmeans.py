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
    ############################################################################
    # EJERCICIO 2: SE DEBE COMPLETAR EL CÓDIGO POR EL ESTUDIANTE.
    # AYUDA: construya una matriz de distancias (N, k): la columna j es la distancia
    #        (al cuadrado basta) de todos los puntos al centroide C[j]:
    #            np.sum((X - C[j]) ** 2, axis=1)
    #        Después, np.argmin(..., axis=1) da el centroide más cercano de cada fila.
    #        Evite recorrer los N puntos con un bucle: con 100.000 puntos es muy lento.
    ############################################################################
    raise NotImplementedError('EJERCICIO 2 sin completar: asignar()')


def actualizar(X, etiquetas, C):
    """
    PASO 2 · ACTUALIZACIÓN. Devuelve los nuevos centroides (k, d): el centroide j pasa
    a ser la MEDIA de los puntos con etiqueta j. Si un grupo se ha quedado vacío,
    conserva su centroide anterior C[j].
    """
    ############################################################################
    # EJERCICIO 2: SE DEBE COMPLETAR EL CÓDIGO POR EL ESTUDIANTE.
    # AYUDA: X[etiquetas == j] son los puntos del grupo j; .mean(axis=0) es su media.
    ############################################################################
    raise NotImplementedError('EJERCICIO 2 sin completar: actualizar()')


def inercia(X, etiquetas, C):
    """
    Suma de las distancias AL CUADRADO de cada punto a su centroide. Es lo que k-means
    intenta hacer pequeño: cuanto menor, más compactos son los grupos.
    """
    ############################################################################
    # EJERCICIO 2: SE DEBE COMPLETAR EL CÓDIGO POR EL ESTUDIANTE.
    # AYUDA: C[etiquetas] es una matriz (N, d) con el centroide de cada punto.
    ############################################################################
    raise NotImplementedError('EJERCICIO 2 sin completar: inercia()')


def kmeans(X, k, n_iter=100, semilla=0, C_inicial=None):
    """
    Agrupa los puntos de X en k grupos. Repite asignar + actualizar hasta que los
    centroides dejan de moverse o se alcanzan n_iter iteraciones.

    Si se da C_inicial, se parte de esos centroides; si no, de k puntos al azar.
    Devuelve (etiquetas, C, inercia, iteraciones realizadas).
    """
    C = inicializar(X, k, semilla) if C_inicial is None else np.array(C_inicial, dtype=float)
    ############################################################################
    # EJERCICIO 2: SE DEBE COMPLETAR EL CÓDIGO POR EL ESTUDIANTE.
    # AYUDA: for it in range(1, n_iter + 1):
    #            asignar -> actualizar -> si np.allclose(C_nuevo, C): parar
    #        Al terminar, vuelva a asignar con los centroides finales y calcule la inercia.
    ############################################################################
    raise NotImplementedError('EJERCICIO 2 sin completar: kmeans()')
