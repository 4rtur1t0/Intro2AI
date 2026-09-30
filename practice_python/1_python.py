"""
    PRÁCTICA 1. ENTORNO DE TRABAJO Y REPASO DE PYTHON.

    PARTE 2: EL LENGUAJE PYTHON.

    Cuatro ejercicios del guión (3.1 a 3.4). Cada ejercicio es una función:
    lea el enunciado en el docstring, complete el cuerpo donde pone
    "EJERCICIO: SE DEBE COMPLETAR EL CÓDIGO POR EL ESTUDIANTE" y ejecute el script.
    El programa principal imprime lo que devuelve cada función junto al valor esperado.

    Ejecute el script DESDE ESTE DIRECTORIO:   python 1_python.py
"""


# ---------------------------------------------------------------------------------------
# 3.1 Booleanos y condicionales
# ---------------------------------------------------------------------------------------
def es_bisiesto(anio):
    """
    Un año es bisiesto si es divisible por 4, salvo los divisibles por 100, que solo lo
    son si además son divisibles por 400.
        es_bisiesto(2024)  ->  True
        es_bisiesto(1900)  ->  False
        es_bisiesto(2000)  ->  True
    """
    # EJERCICIO: SE DEBE COMPLETAR EL CÓDIGO POR EL ESTUDIANTE
    return None


# ---------------------------------------------------------------------------------------
# 3.2 Listas y tuplas
# ---------------------------------------------------------------------------------------
def sin_extremos(valores):
    """
    Devuelve una lista NUEVA con los valores ordenados de menor a mayor y sin el mínimo ni
    el máximo. La lista original NO debe modificarse.
        sin_extremos([7, 2, 9, 4, 5])  ->  [4, 5, 7]
        sin_extremos([3, 1])           ->  []
    """
    # EJERCICIO: SE DEBE COMPLETAR EL CÓDIGO POR EL ESTUDIANTE
    return None




# ---------------------------------------------------------------------------------------
# 3.3 Diccionarios
# ---------------------------------------------------------------------------------------
def distancia_ruta(ruta, red):
    """
    'red' es un diccionario de diccionarios: red[ciudad][vecina] es la distancia en km.
    Devuelve la distancia total de la ruta (lista de ciudades consecutivas).
        distancia_ruta(['Cuenca', 'Madrid', 'Toledo'], RED)  ->  237
    """
    # EJERCICIO: SE DEBE COMPLETAR EL CÓDIGO POR EL ESTUDIANTE
    return None


RED = {'Madrid': {'Toledo': 72, 'Cuenca': 165, 'Guadalajara': 60},
       'Toledo': {'Madrid': 72, 'Ciudad Real': 120},
       'Cuenca': {'Madrid': 165},
       'Guadalajara': {'Madrid': 60},
       'Ciudad Real': {'Toledo': 120}}


# ---------------------------------------------------------------------------------------
# 3.4 Módulos y librerías
# ---------------------------------------------------------------------------------------
def distancia_euclidea(p, q):
    """
    Distancia euclídea entre dos puntos dados como listas, sqrt(sum((p_i - q_i)^2)),
    usando sqrt del módulo math y un bucle sobre zip(p, q).
        distancia_euclidea([1, 2, 3], [4, 6, 3])  ->  5.0
    """
    # EJERCICIO: SE DEBE COMPLETAR EL CÓDIGO POR EL ESTUDIANTE
    return None


# ---------------------------------------------------------------------------------------
# Programa principal: ejecuta cada ejercicio y muestra el resultado junto al esperado
# ---------------------------------------------------------------------------------------
def mostrar(nombre, obtenido, esperado):
    print('  ' + nombre)
    print('      obtenido: %r' % (obtenido,))
    print('      esperado: %r' % (esperado,))


if __name__ == '__main__':

    print('3.1 Booleanos y condicionales')
    mostrar('es_bisiesto(2024)', es_bisiesto(2024), True)
    mostrar('es_bisiesto(1900)', es_bisiesto(1900), False)
    mostrar('es_bisiesto(2000)', es_bisiesto(2000), True)


    print('3.2 Listas y tuplas')
    original = [7, 2, 9, 4, 5]
    mostrar('sin_extremos([7, 2, 9, 4, 5])', sin_extremos(original), [4, 5, 7])
    mostrar('  ... y la original sigue siendo', original, [7, 2, 9, 4, 5])
  

    print('3.3 Diccionarios')
    mostrar("distancia_ruta(['Cuenca', 'Madrid', 'Toledo'], RED)",
            distancia_ruta(['Cuenca', 'Madrid', 'Toledo'], RED), 237)
    mostrar("distancia_ruta(['Guadalajara', 'Madrid', 'Toledo', 'Ciudad Real'], RED)",
            distancia_ruta(['Guadalajara', 'Madrid', 'Toledo', 'Ciudad Real'], RED), 252)

    print('3.4 Módulos')
    mostrar('distancia_euclidea([1, 2, 3], [4, 6, 3])', distancia_euclidea([1, 2, 3], [4, 6, 3]), 5.0)
