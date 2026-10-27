"""
    El suelo según las etiquetas de SemanticKITTI y las métricas para evaluar
    (YA PROGRAMADO: no hay que tocar nada).

    Las etiquetas NO se usan para agrupar: los algoritmos no las ven. Solo sirven para MEDIR
    cómo de bien lo han hecho (ejercicio 3) y para quitar el suelo antes de buscar objetos
    (ejercicio 4). En un coche real no hay etiquetas: el suelo se quitaría con un filtro
    geométrico, como un plano ajustado con RANSAC o el filtro CSF (simulación de una tela).
"""
import numpy as np

# calzada, aparcamiento, acera, otro suelo, marca vial y terreno (césped, tierra)
CLASES_SUELO = [40, 44, 48, 49, 60, 72]
# sin etiqueta y valores atípicos: no cuentan ni como suelo ni como objeto
CLASES_DUDOSAS = [0, 1]
# el sensor va a 1,73 m del suelo: z = 0,8 m son unos 2,5 m de altura
Z_MAX_OBJETOS = 0.8


def es_suelo(clase):
    """Vector booleano (N,): True para los puntos etiquetados como suelo."""
    return np.isin(clase, CLASES_SUELO)


def es_objeto(nube, clase, z_max=Z_MAX_OBJETOS):
    """
    Vector booleano (N,): True para los puntos que NO son suelo (ni dudosos) y están por
    debajo de z_max. Por encima quedan copas de árboles y aleros, que no interesan para
    separar objetos y que, vistos desde arriba, unirían cosas que están debajo.
    """
    return ~es_suelo(clase) & ~np.isin(clase, CLASES_DUDOSAS) & (nube[:, 2] < z_max)


def prediccion_de_suelo(etiquetas, clase):
    """
    k-means devuelve dos grupos (0 y 1), pero no sabe cuál de ellos es "el suelo". Para
    evaluar un agrupamiento con etiquetas se toma la correspondencia grupo -> clase que más
    puntos acierta (es lo que se hace siempre al comparar un agrupamiento con etiquetas).
    Devuelve un vector booleano (N,): True = el agrupamiento dice que el punto es suelo.
    """
    validos = ~np.isin(clase, CLASES_DUDOSAS)
    suelo = es_suelo(clase)
    prediccion = etiquetas == 0
    if np.sum(prediccion[validos] == suelo[validos]) < np.sum(~prediccion[validos] == suelo[validos]):
        prediccion = ~prediccion
    return prediccion


def matriz_de_confusion(prediccion, clase):
    """
    Matriz de confusión con el SUELO como clase positiva (solo puntos con etiqueta válida):
        TP (verdaderos positivos): suelo que se predice suelo
        FN (falsos negativos):     suelo que se predice objeto
        FP (falsos positivos):     objeto que se predice suelo
        TN (verdaderos negativos): objeto que se predice objeto
    Devuelve (TP, FP, FN, TN).
    """
    validos = ~np.isin(clase, CLASES_DUDOSAS)
    real, pred = es_suelo(clase)[validos], prediccion[validos]
    TP = int(np.sum(real & pred))
    FN = int(np.sum(real & ~pred))
    FP = int(np.sum(~real & pred))
    TN = int(np.sum(~real & ~pred))
    return TP, FP, FN, TN


def metricas(TP, FP, FN, TN):
    """
    Métricas estándar de clasificación binaria a partir de la matriz de confusión:
        accuracy  = (TP + TN) / total     fracción de puntos bien clasificados
        precision = TP / (TP + FP)        de lo que se predice suelo, cuánto lo es
        recall    = TP / (TP + FN)        del suelo real, cuánto se encuentra
        F1-score  = 2 P R / (P + R)       media armónica de precision y recall
        IoU       = TP / (TP + FP + FN)   intersección sobre unión (índice de Jaccard),
                                          la métrica oficial de SemanticKITTI
    Devuelve un diccionario con las cinco, entre 0 y 1.
    """
    precision = TP / (TP + FP) if TP + FP > 0 else 0.0
    recall = TP / (TP + FN) if TP + FN > 0 else 0.0
    return {'accuracy': (TP + TN) / (TP + FP + FN + TN),
            'precision': precision,
            'recall': recall,
            'F1-score': 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0,
            'IoU': TP / (TP + FP + FN) if TP + FP + FN > 0 else 0.0}


def imprimir_evaluacion(TP, FP, FN, TN):
    """Imprime la matriz de confusión y las métricas en porcentaje."""
    print('                      predicho suelo   predicho objeto')
    print('     real suelo      %15d %17d' % (TP, FN))
    print('     real objeto     %15d %17d' % (FP, TN))
    print('     ' + '   '.join('%s %.1f %%' % (k, 100 * v) for k, v in metricas(TP, FP, FN, TN).items()))
