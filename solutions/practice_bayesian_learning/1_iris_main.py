"""
    EJERCICIO 1: NAÏVE BAYES GAUSSIANO, CON LAS FLORES IRIS.

    Completar: naive_bayes/gaussiano.py: entrenar_naive_bayes en el apartado a), y
    log_densidad_gaussiana y calcular_puntuaciones en el b).

        a) entrena con las 114 flores de entrenamiento, escribe la media y la desviación
           típica de cada atributo en cada variedad y dibuja sus campanas (la figura 2
           del guion, que se guarda en figures/ej1_campanas.png)
        b) clasifica la primera flor de prueba, sumando los logaritmos atributo a atributo
        c) clasifica las 36 flores de prueba (tasa de acierto y matriz de confusión) y
           compara con GaussianNB de scikit-learn

    Ejecútelo desde este directorio:   python 1_iris_main.py
"""
import numpy as np
from sklearn.naive_bayes import GaussianNB      # solo para comparar, en c)
from naive_bayes.gaussiano import (entrenar_naive_bayes, log_densidad_gaussiana,
                                   calcular_puntuaciones, predecir_naive_bayes)
from naive_bayes.datos import cargar_iris, train_test_split
from naive_bayes.evaluacion import matriz_de_confusion, escribir_matriz_de_confusion
from herramientas import comprobar, intentar, resumen, figura_campanas

VARIEDADES = ['setosa', 'versicolor', 'virginica']      # las clases 0, 1 y 2
NUM_CLASES = 3
ATRIBUTOS = ['largo del sépalo', 'ancho del sépalo', 'largo del pétalo', 'ancho del pétalo']


def las_flores():
    """Las 150 flores (50 de cada variedad): 114 de entrenamiento y 36 de prueba."""
    X, y = cargar_iris()
    return train_test_split(X, y, test_size=0.25, seed=0)


def entrenar(X_train, X_test, y_train, y_test):
    medias, desviaciones, priors = entrenar_naive_bayes(X_train, y_train, NUM_CLASES)
    print('P(c): ' + '   '.join('P(%s) = %.4f' % (VARIEDADES[c], priors[c])
                                for c in range(NUM_CLASES)))
    for titulo, tabla in (('MEDIAS (cm)', medias), ('DESVIACIONES TÍPICAS (cm)', desviaciones)):
        print('%-26s' % titulo + ''.join('%12s' % variedad for variedad in VARIEDADES))
        for i, atributo in enumerate(ATRIBUTOS):
            print('   %-23s' % atributo + ''.join('%12.4f' % tabla[c][i]
                                                 for c in range(NUM_CLASES)))
    comprobar('P(c) de cada variedad', [round(p, 4) for p in priors], [0.3333] * 3,
              'P(c) es la proporción de flores de entrenamiento de la variedad c.')
    comprobar('medias del largo del pétalo', [round(m[2], 4) for m in medias],
              [1.4921, 4.2316, 5.5316],
              'La media de cada atributo, solo con las flores de cada variedad: '
              'X_train[y_train == c].')
    comprobar('desviaciones típicas del largo del pétalo', [round(d[2], 4) for d in desviaciones],
              [0.166, 0.4554, 0.5171],
              'np.std calcula la desviación típica (np.var, la varianza), y divide entre n: '
              'no le pase ddof=1.')
    figura_campanas(X_train, y_train, medias, desviaciones, VARIEDADES, ATRIBUTOS,
                    ruta='figures/ej1_campanas.png')


def una_flor(X_train, X_test, y_train, y_test):
    medias, desviaciones, priors = entrenar_naive_bayes(X_train, y_train, NUM_CLASES)
    x = X_test[0]
    print('La primera flor de prueba, una %s: %s cm' % (VARIEDADES[y_test[0]], x))
    logs = [[log_densidad_gaussiana(x[i], medias[c][i], desviaciones[c][i])
             for i in range(len(x))] for c in range(NUM_CLASES)]
    print('log P(c) y log P(x_i|c) de cada atributo (x_1, el largo del sépalo, ..., x_4, el '
          'ancho del pétalo):')
    print('%-12s%10s%9s%9s%9s%9s' % ('', 'log P(c)', 'x_1', 'x_2', 'x_3', 'x_4'))
    for c in range(NUM_CLASES):
        print('%-12s%10.3f' % (VARIEDADES[c], np.log(priors[c]))
              + ''.join('%9.3f' % v for v in logs[c]))
    comprobar('log P(x_i|setosa) de los cuatro atributos', [round(v, 3) for v in logs[0]],
              [0.143, 0.026, 0.723, 1.145],
              'Es la fórmula de log_densidad_gaussiana, con np.log y np.pi. Ojo: la '
              'fórmula lleva el CUADRADO de la desviación típica.')

    puntuaciones = calcular_puntuaciones(x, medias, desviaciones, priors, NUM_CLASES)
    print('La puntuación de cada variedad, log P(c) más la suma de sus log P(x_i|c):')
    print('   ' + '   '.join('%s %.3f' % (VARIEDADES[c], puntuaciones[c])
                             for c in range(NUM_CLASES))
          + '   ->  %s' % VARIEDADES[predecir_naive_bayes(x, medias, desviaciones, priors,
                                                          NUM_CLASES)])
    comprobar('puntuación de cada variedad: log P(c) + suma de log P(x_i|c)',
              [round(puntuaciones[c], 3) for c in range(NUM_CLASES)], [0.939, -40.42, -59.571],
              'Para cada clase, log P(c) más los log P(x_i|c) de los cuatro atributos, cada '
              'uno con la media y la desviación típica de su atributo en esa clase.')


def prueba(X_train, X_test, y_train, y_test):
    medias, desviaciones, priors = entrenar_naive_bayes(X_train, y_train, NUM_CLASES)
    y_pred = np.array([predecir_naive_bayes(x, medias, desviaciones, priors, NUM_CLASES)
                       for x in X_test])
    y_pred_train = np.array([predecir_naive_bayes(x, medias, desviaciones, priors, NUM_CLASES)
                             for x in X_train])
    acierto = np.mean(y_pred == y_test)
    print('Tasa de acierto: %.4f en las 114 flores de entrenamiento y %.4f en las 36 de '
          'prueba.' % (np.mean(y_pred_train == y_train), acierto))
    matriz = matriz_de_confusion(y_test, y_pred, NUM_CLASES)
    escribir_matriz_de_confusion(matriz, VARIEDADES)
    comprobar('tasa de acierto en las flores de prueba', round(acierto, 4), 0.9444)
    comprobar('matriz de confusión', matriz.tolist(), [[12, 0, 0], [0, 11, 1], [0, 1, 11]])

    referencia = GaussianNB().fit(X_train, y_train)
    iguales = bool(np.all(referencia.predict(X_test) == y_pred))
    print('GaussianNB de scikit-learn, con las mismas flores: ¿las mismas predicciones? %s'
          % iguales)
    comprobar('las mismas predicciones que GaussianNB', iguales, True)


if __name__ == "__main__":
    datos = las_flores()
    intentar('a) ENTRENAR: MEDIAS Y DESVIACIONES TÍPICAS', lambda: entrenar(*datos))
    intentar('b) UNA FLOR, PASO A PASO', lambda: una_flor(*datos))
    intentar('c) LAS 36 FLORES DE PRUEBA', lambda: prueba(*datos))
    resumen()
