"""
    La matriz de confusión, la precisión y la exhaustividad, programadas desde cero.

    Se dan programadas, pero conviene leerlas: la matriz de confusión es solo una tabla
    de conteos, y la precisión y la exhaustividad son dos cocientes sobre esa tabla.
"""
import numpy as np


def matriz_de_confusion(y_real, y_predicho, num_clases):
    """
    En la fila de la clase real y la columna de la clase predicha, cuántos ejemplos hay.
    """
    matriz = np.zeros((num_clases, num_clases), dtype=int)
    for real, predicho in zip(y_real, y_predicho):
        matriz[real][predicho] += 1
    return matriz


def escribir_matriz_de_confusion(matriz, nombres_clases):
    print('Matriz de confusión (filas: clase real; columnas: clase predicha):')
    print('%12s' % '' + ''.join('%12s' % nombre for nombre in nombres_clases))
    for c, nombre in enumerate(nombres_clases):
        print('%12s' % nombre + ''.join('%12d' % n for n in matriz[c]))


def precision_exhaustividad(matriz):
    """
    Las dos métricas de la clase 1 (el spam), con VP los verdaderos positivos (spam
    marcado como spam), FP los falsos positivos (ham marcado como spam) y FN los falsos
    negativos (spam que se cuela como ham):

        precisión     = VP / (VP + FP): de los que se han marcado como spam, cuántos lo eran
        exhaustividad = VP / (VP + FN): de los spam que había, cuántos se han encontrado
    """
    vp, fp, fn = matriz[1][1], matriz[0][1], matriz[1][0]
    precision = vp / (vp + fp) if vp + fp > 0 else 0.0
    exhaustividad = vp / (vp + fn) if vp + fn > 0 else 0.0
    return float(precision), float(exhaustividad)
