"""
    EJERCICIO 3: NAÏVE BAYES MULTINOMIAL, UN FILTRO DE SPAM.

    Completar: naive_bayes/multinomial.py: entrenar_naive_bayes_multinomial en el
    apartado a), y calcular_puntuaciones_multinomial en el b). Necesita el ejercicio 2
    hecho: los SMS se convierten en números con su bolsa de palabras.

        a) entrena el filtro con los 4.180 SMS de entrenamiento y escribe P(c) y, para la
           palabra 'prize', cuántas veces aparece en cada clase y log P(prize|c)
        b) clasifica el primer SMS de prueba y después los 1.394 (tasa de acierto, matriz
           de confusión, precisión y exhaustividad), y compara con MultinomialNB de
           scikit-learn
        c) escribe las palabras más frecuentes y las más indicativas de cada clase
        d) no está aquí: intente engañar al filtro con   python filtro.py

    Ejecútelo desde este directorio:   python 3_spam_main.py
"""
import numpy as np
from sklearn.naive_bayes import MultinomialNB      # solo para comparar, en a) y b)
from naive_bayes.bolsa_de_palabras import (construir_vocabulario, vectorizar, tokenizar,
                                           palabras_del_vocabulario)
from naive_bayes.multinomial import (entrenar_naive_bayes_multinomial,
                                     calcular_puntuaciones_multinomial,
                                     predecir_naive_bayes_multinomial,
                                     palabras_mas_frecuentes, palabras_mas_indicativas)
from naive_bayes.datos import cargar_sms, train_test_split
from naive_bayes.evaluacion import (matriz_de_confusion, escribir_matriz_de_confusion,
                                    precision_exhaustividad)
from herramientas import comprobar, intentar, resumen, miles

CLASES = ['ham', 'spam']        # las clases 0 y 1
NUM_CLASES = 2


def los_sms():
    """Los SMS de entrenamiento y de prueba, convertidos en números con la bolsa de palabras."""
    mensajes, y = cargar_sms()
    textos_train, textos_test, y_train, y_test = train_test_split(mensajes, y,
                                                                  test_size=0.25, seed=0)
    vocabulario = construir_vocabulario(textos_train)
    X_train = vectorizar(textos_train, vocabulario)
    X_test = vectorizar(textos_test, vocabulario)
    return vocabulario, textos_test, X_train, X_test, y_train, y_test


def entrenar():
    vocabulario, textos_test, X_train, X_test, y_train, y_test = los_sms()
    log_priors, log_prob_palabras = entrenar_naive_bayes_multinomial(X_train, y_train,
                                                                     NUM_CLASES, alpha=1.0)
    priors = np.exp(log_priors)
    print('P(c): P(ham) = %.4f   P(spam) = %.4f' % (priors[0], priors[1]))
    i = vocabulario['prize']
    for c in range(NUM_CLASES):
        X_c = X_train[y_train == c]
        print("'prize' sale %d veces en los SMS %s, que suman %s palabras: "
              "log P(prize|%s) = %.4f" % (X_c[:, i].sum(), CLASES[c], miles(X_c.sum()),
                                          CLASES[c], log_prob_palabras[c][i]))
    comprobar('P(ham) y P(spam)', [round(p, 4) for p in priors], [0.866, 0.134],
              'log P(c) es el logaritmo de la proporción de SMS de la clase c.')
    comprobar('log P(prize|ham) y log P(prize|spam)',
              [round(log_prob_palabras[c][i], 4) for c in range(NUM_CLASES)],
              [-10.9097, -5.5732],
              'Arriba, N(palabra, c) + alpha; abajo, N(c) + alpha por el tamaño del '
              'vocabulario.')

    # con otro suavizado, sus log P(palabra|c) deben ser los de MultinomialNB
    log_priors, log_prob_palabras = entrenar_naive_bayes_multinomial(X_train, y_train,
                                                                     NUM_CLASES, alpha=0.5)
    referencia = MultinomialNB(alpha=0.5).fit(X_train, y_train)
    comprobar('con alpha = 0,5, los mismos log P(palabra|c) que MultinomialNB',
              bool(np.allclose(log_prob_palabras, referencia.feature_log_prob_)), True,
              'Abajo, alpha multiplica al tamaño del vocabulario: con alpha = 1 no se nota '
              'si falta.')


def prueba():
    vocabulario, textos_test, X_train, X_test, y_train, y_test = los_sms()
    log_priors, log_prob_palabras = entrenar_naive_bayes_multinomial(X_train, y_train,
                                                                     NUM_CLASES, alpha=1.0)
    puntuacion = calcular_puntuaciones_multinomial(X_test[0], log_priors, log_prob_palabras)
    print('El primer SMS de prueba, %s: %r' % (CLASES[y_test[0]], textos_test[0]))
    print('   sus palabras del vocabulario: %s'
          % ', '.join(p for p in tokenizar(textos_test[0]) if p in vocabulario))
    print('   puntuación(ham) = %.3f   puntuación(spam) = %.3f   ->  %s'
          % (puntuacion[0], puntuacion[1], CLASES[int(np.argmax(puntuacion))]))
    comprobar('puntuaciones del primer SMS de prueba', [round(p, 3) for p in puntuacion],
              [-120.816, -143.235],
              'Para cada clase, log P(c) más, palabra a palabra del vocabulario, su conteo '
              'en el mensaje por su log P(palabra|c).')

    print('Clasificando los %s SMS de prueba, uno a uno y palabra a palabra...'
          % miles(len(y_test)))
    y_pred = np.array([predecir_naive_bayes_multinomial(x, log_priors, log_prob_palabras)
                       for x in X_test])
    acierto = np.mean(y_pred == y_test)
    print('Tasa de acierto en los SMS de prueba: %.4f' % acierto)
    matriz = matriz_de_confusion(y_test, y_pred, NUM_CLASES)
    escribir_matriz_de_confusion(matriz, CLASES)
    precision, exhaustividad = precision_exhaustividad(matriz)
    print('Spam: precisión = %.4f   exhaustividad = %.4f' % (precision, exhaustividad))
    comprobar('tasa de acierto en los SMS de prueba', round(acierto, 4), 0.9792)
    comprobar('matriz de confusión', matriz.tolist(), [[1199, 8], [21, 166]])
    comprobar('precisión y exhaustividad del spam',
              [round(precision, 4), round(exhaustividad, 4)], [0.954, 0.8877])

    referencia = MultinomialNB(alpha=1.0).fit(X_train, y_train)
    iguales = bool(np.all(referencia.predict(X_test) == y_pred))
    print('MultinomialNB de scikit-learn, con la misma matriz de conteos: ¿las mismas '
          'predicciones? %s' % iguales)
    comprobar('las mismas predicciones que MultinomialNB', iguales, True)


def palabras():
    vocabulario, textos_test, X_train, X_test, y_train, y_test = los_sms()
    log_priors, log_prob_palabras = entrenar_naive_bayes_multinomial(X_train, y_train,
                                                                     NUM_CLASES, alpha=1.0)
    lista = palabras_del_vocabulario(vocabulario)
    print('Las más frecuentes de cada clase (mayor P(palabra|c)):')
    for c, frecuentes in enumerate(palabras_mas_frecuentes(log_prob_palabras, lista, n=10)):
        print('   %-4s: %s' % (CLASES[c], ', '.join(frecuentes)))
    spam, ham = palabras_mas_indicativas(log_prob_palabras, lista, n=10)
    print('Las más indicativas de cada clase (mayor diferencia entre log P(palabra|spam) y '
          'log P(palabra|ham)):')
    print('   spam: %s' % ', '.join(spam))
    print('   ham : %s' % ', '.join(ham))
    comprobar('las tres palabras más indicativas de spam', spam[:3], ['claim', 'prize', '150p'])


if __name__ == "__main__":
    intentar('a) ENTRENAR EL FILTRO', entrenar)
    intentar('b) LOS 1.394 SMS DE PRUEBA', prueba)
    intentar('c) LAS PALABRAS EN LAS QUE SE FIJA', palabras)
    resumen()
