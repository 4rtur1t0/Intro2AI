"""
    Carga de los datos y partición en entrenamiento y prueba.

    Se dan programadas: no forman parte de los ejercicios. Aun así, conviene leer
    train_test_split, que hace lo mismo que la de scikit-learn: baraja los ejemplos y
    aparta una parte para la prueba, con la misma proporción de cada clase en las dos
    partes.
"""
import os
import numpy as np
from sklearn.datasets import load_iris

# el fichero de los SMS, en data/: se encuentra se ejecute el programa desde donde se ejecute
SMS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data',
                   'SMSSpamCollection.tsv')


def cargar_iris():
    """
    Las 150 flores iris. Devuelve (X, y): X, un array (150, 4) con sus cuatro medidas en
    cm, e y, un array con su variedad: 0 es setosa; 1, versicolor, y 2, virginica.
    """
    iris = load_iris()
    return iris.data, iris.target


def cargar_sms(ruta=SMS):
    """
    Lee el fichero SMSSpamCollection.tsv: una etiqueta (ham o spam) y un mensaje por
    línea, separados por un tabulador. Devuelve (mensajes, y): la lista de los mensajes y
    un array con su clase, 0 si es ham (legítimo) y 1 si es spam.
    """
    mensajes = []
    y = []
    with open(ruta, 'r', encoding='utf-8') as f:
        for linea in f:
            linea = linea.rstrip('\n')
            if len(linea) == 0:
                continue
            etiqueta, mensaje = linea.split('\t', 1)
            mensajes.append(mensaje)
            y.append(1 if etiqueta == 'spam' else 0)
    return mensajes, np.array(y)


def train_test_split(X, y, test_size=0.25, seed=0):
    """
    Divide los datos en entrenamiento y prueba, manteniendo la proporción de cada clase
    en las dos partes (partición estratificada). Es importante cuando las clases están
    descompensadas: solo el 13 % de los SMS son spam, y una partición desafortunada
    podría dejar muy pocos en el entrenamiento.

    :param X: array de numpy o lista de ejemplos.
    :param y: array de numpy o lista de clases.
    :param test_size: proporción de ejemplos de cada clase que se reservan para la prueba.
    :param seed: semilla del generador aleatorio, para que la partición sea reproducible.
    :return: X_train, X_test, y_train, y_test
    """
    X = np.asarray(X)
    y = np.asarray(y)
    generador = np.random.RandomState(seed)
    indices_train = []
    indices_test = []
    for c in np.unique(y):
        grupo = np.flatnonzero(y == c)                      # los ejemplos de la clase c
        grupo = grupo[generador.permutation(len(grupo))]    # barajados
        n_test = int(round(test_size * len(grupo)))
        indices_test.append(grupo[:n_test])
        indices_train.append(grupo[n_test:])
    indices_train = np.concatenate(indices_train)
    indices_test = np.concatenate(indices_test)
    # se vuelven a barajar para que las clases no queden ordenadas por bloques
    indices_train = indices_train[generador.permutation(len(indices_train))]
    indices_test = indices_test[generador.permutation(len(indices_test))]
    return X[indices_train], X[indices_test], y[indices_train], y[indices_test]
