"""
    PRÁCTICA 1. ENTORNO DE TRABAJO Y REPASO DE PYTHON.

    PARTE 4: NUMPY.

    Un ejercicio por cada apartado del guión (4.1 a 4.4). Lea el enunciado en el
    docstring, complete el cuerpo donde pone "EJERCICIO: SE DEBE COMPLETAR EL CÓDIGO POR
    EL ESTUDIANTE" y ejecute el script. Todos los ejercicios se resuelven SIN bucles.

    Ejecute el script DESDE ESTE DIRECTORIO:   python 2_numpy.py
"""
import numpy as np


# ---------------------------------------------------------------------------------------
# 4.1 Crear arrays, forma y ejes
# ---------------------------------------------------------------------------------------
def matriz_y_sumas():
    """
    Crea la matriz A de 4 filas y 5 columnas con los enteros 0, 1, ..., 19 (np.arange y
    reshape) y devuelve (A, suma de cada fila, suma de cada columna).
        A = [[ 0  1  2  3  4]
             [ 5  6  7  8  9]
             [10 11 12 13 14]
             [15 16 17 18 19]]
        suma por filas    ->  [10 35 60 85]        (forma (4,))
        suma por columnas ->  [30 34 38 42 46]     (forma (5,))
    """
    # EJERCICIO: SE DEBE COMPLETAR EL CÓDIGO POR EL ESTUDIANTE
    return None


# ---------------------------------------------------------------------------------------
# 4.2 Operaciones vectorizadas y broadcasting
# ---------------------------------------------------------------------------------------
def distancia_euclidea(p, q):
    """
    La misma distancia euclídea de la parte anterior, sqrt(sum((p_i - q_i)^2)), pero ahora
    con p y q como arrays de numpy y en UNA sola línea, sin bucles: la resta, el cuadrado y
    np.sqrt operan elemento a elemento.
        distancia_euclidea(np.array([1, 2, 3]), np.array([4, 6, 3]))  ->  5.0
    """
    # EJERCICIO: SE DEBE COMPLETAR EL CÓDIGO POR EL ESTUDIANTE
    return None


# ---------------------------------------------------------------------------------------
# 4.3 Máscaras booleanas
# ---------------------------------------------------------------------------------------
def filas_de_la_clase(X, y, c):
    """
    X es una matriz con un ejemplo por fila; y es un vector con la clase (un entero) de
    cada fila. Devuelve las filas de X cuya clase es c, con una máscara booleana.
        X = [[1, 2], [3, 4], [5, 6], [7, 8]],  y = [0, 1, 0, 1]
        filas_de_la_clase(X, y, 1)  ->  [[3 4]
                                         [7 8]]
    """
    # EJERCICIO: SE DEBE COMPLETAR EL CÓDIGO POR EL ESTUDIANTE
    return None


def contar_clases(y):
    """
    Devuelve (clases, conteos): las clases distintas que aparecen en y, ordenadas, y cuántas
    veces aparece cada una. Use np.unique con return_counts=True.
        contar_clases(np.array([2, 0, 1, 0, 2, 0]))  ->  ([0 1 2], [3 1 2])
    """
    # EJERCICIO: SE DEBE COMPLETAR EL CÓDIGO POR EL ESTUDIANTE
    return None


def recortar_negativos(A):
    """
    Devuelve una COPIA de A con los valores negativos sustituidos por 0, y cuántos había.
    La matriz original no debe cambiar.
        recortar_negativos(np.array([[1, -2], [-3, 4]]))  ->  ([[1 0]
                                                                [0 4]],  2)
    """
    # EJERCICIO: SE DEBE COMPLETAR EL CÓDIGO POR EL ESTUDIANTE
    return None


# ---------------------------------------------------------------------------------------
# 4.4 Todo junto: medias por clase
# ---------------------------------------------------------------------------------------
def medias_por_clase(X, y):
    """
    Devuelve la matriz con una fila por clase y una columna por atributo: la media de cada
    atributo entre los ejemplos de esa clase. Se admite un único bucle sobre las clases.
        X = [[1, 2], [3, 4], [5, 6], [7, 8]],  y = [0, 1, 0, 1]
        medias_por_clase(X, y)  ->  [[3. 4.]
                                     [5. 6.]]
    """
    # EJERCICIO: SE DEBE COMPLETAR EL CÓDIGO POR EL ESTUDIANTE
    return None


def producto_matrices(A, B):
    """
    Devuelve (A * B, A @ B): el producto elemento a elemento y el producto de matrices.
        A = [[1, 2], [3, 4]],  B = [[1, 0], [0, 1]]
        A * B  ->  [[1 0]        A @ B  ->  [[1 2]
                    [0 4]]                   [3 4]]
    """
    # EJERCICIO: SE DEBE COMPLETAR EL CÓDIGO POR EL ESTUDIANTE
    return None


# ---------------------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------------------
if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)

    print('4.1 matriz_y_sumas()')
    resultado = matriz_y_sumas()
    if resultado is None:
        print('  (pendiente)')
    else:
        A, filas, columnas = resultado
        print(A)
        print('  suma por filas    :', filas, '   esperado: [10 35 60 85]')
        print('  suma por columnas :', columnas, '   esperado: [30 34 38 42 46]')

    print('4.2 distancia_euclidea(p, q)')
    print('  obtenido:', distancia_euclidea(np.array([1, 2, 3]), np.array([4, 6, 3])),
          '   esperado: 5.0')

    print('4.3 máscaras')
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 1, 0, 1])
    print('  filas_de_la_clase(X, y, 1):\n', filas_de_la_clase(X, y, 1), '   esperado: [[3 4] [7 8]]')
    print('  contar_clases([2, 0, 1, 0, 2, 0]):', contar_clases(np.array([2, 0, 1, 0, 2, 0])),
          '   esperado: ([0 1 2], [3 1 2])')
    A = np.array([[1, -2], [-3, 4]])
    print('  recortar_negativos(A):', recortar_negativos(A), '   esperado: ([[1 0] [0 4]], 2)')
    print('  ... y A sigue siendo:', A.tolist(), '   esperado: [[1, -2], [-3, 4]]')

    print('4.4 medias_por_clase(X, y)')
    print(medias_por_clase(X, y), '   esperado: [[3. 4.] [5. 6.]]')
    A = np.array([[1, 2], [3, 4]])
    B = np.eye(2, dtype=int)
    print('  producto_matrices(A, I):', producto_matrices(A, B),
          '   esperado: ([[1 0] [0 4]], [[1 2] [3 4]])')
