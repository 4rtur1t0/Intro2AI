"""
    EJERCICIO 2: LA BOLSA DE PALABRAS.

    Completar: naive_bayes/bolsa_de_palabras.py (construir_vocabulario y vectorizar).

        a) construye el vocabulario y la matriz de conteos de tres mensajes cortos, y
           convierte en un vector un cuarto mensaje, con una palabra que no está en el
           vocabulario
        b) lo mismo con los 4.180 SMS de entrenamiento, y compara la matriz con la de
           CountVectorizer de scikit-learn

    Ejecútelo desde este directorio:   python 2_bolsa_de_palabras_main.py
"""
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer    # solo para comparar, en b)
from naive_bayes.bolsa_de_palabras import (construir_vocabulario, vectorizar,
                                           palabras_del_vocabulario)
from naive_bayes.datos import cargar_sms, train_test_split
from herramientas import comprobar, intentar, resumen, miles

MENSAJES = ['Win a FREE prize now', 'Call me now', 'I win, you win']
NUEVO = 'Call now to win a prize'


def tres_mensajes():
    vocabulario = construir_vocabulario(MENSAJES)
    print('Vocabulario: %s' % vocabulario)
    comprobar('vocabulario, por orden alfabético', vocabulario,
              {'call': 0, 'free': 1, 'me': 2, 'now': 3, 'prize': 4, 'win': 5, 'you': 6},
              'Cada palabra distinta una sola vez, y ordenadas: sorted(...). La primera tiene '
              'el índice 0.')
    X = vectorizar(MENSAJES, vocabulario)
    print('%-28s' % '' + ''.join('%7s' % palabra
                                 for palabra in palabras_del_vocabulario(vocabulario)))
    for mensaje, fila in zip(MENSAJES, X):
        print('%-28s' % repr(mensaje) + ''.join('%7d' % n for n in fila))
    comprobar('matriz de conteos de los tres mensajes', X.tolist(),
              [[0, 1, 0, 1, 1, 1, 0], [1, 0, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 2, 1]],
              'En la fila de cada mensaje, 1 más en la columna de una palabra cada vez que '
              'aparece.')
    x = vectorizar([NUEVO], vocabulario)
    print('%-28s' % repr(NUEVO) + ''.join('%7d' % n for n in x[0]))
    comprobar('vector del mensaje nuevo, %r' % NUEVO, x[0].tolist(), [1, 0, 0, 1, 1, 1, 0],
              '«to» no está en el vocabulario: hay que ignorarla.')


def los_sms():
    mensajes, y = cargar_sms()
    textos_train, textos_test, y_train, y_test = train_test_split(mensajes, y,
                                                                  test_size=0.25, seed=0)
    vocabulario = construir_vocabulario(textos_train)
    X = vectorizar(textos_train, vocabulario)
    ceros = np.mean(X == 0)
    print('%s SMS, %s de entrenamiento y %s de prueba. Con los de entrenamiento:'
          % (miles(len(mensajes)), miles(len(textos_train)), miles(len(textos_test))))
    print('   vocabulario: %s palabras' % miles(len(vocabulario)))
    print('   matriz de conteos: %s x %s, con un %.2f %% de ceros; guardada entera, ocupa '
          '%.1f MB' % (miles(X.shape[0]), miles(X.shape[1]), 100 * ceros, X.nbytes / 1e6))
    comprobar('palabras del vocabulario', len(vocabulario), 7498,
              'El vocabulario se construye solo con los SMS de entrenamiento.')
    comprobar('proporción de ceros de la matriz', round(ceros, 4), 0.9982)

    referencia = CountVectorizer()
    X_ref = referencia.fit_transform(textos_train)        # una matriz dispersa (sparse)
    memoria = (X_ref.data.nbytes + X_ref.indices.nbytes + X_ref.indptr.nbytes) / 1e6
    print('CountVectorizer de scikit-learn la guarda como una matriz dispersa: ocupa %.1f MB'
          % memoria)
    iguales = (palabras_del_vocabulario(vocabulario) == list(referencia.get_feature_names_out())
               and np.array_equal(X, X_ref.toarray()))
    comprobar('el mismo vocabulario y la misma matriz que CountVectorizer', iguales, True)


if __name__ == "__main__":
    intentar('a) TRES MENSAJES CORTOS', tres_mensajes)
    intentar('b) LOS 4.180 SMS DE ENTRENAMIENTO', los_sms)
    resumen()
