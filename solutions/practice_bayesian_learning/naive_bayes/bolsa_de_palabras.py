"""
    EJERCICIO 2 · DE TEXTO A NÚMEROS: LA BOLSA DE PALABRAS (BAG OF WORDS).

    Un clasificador no sabe leer: necesita vectores de números. La bolsa de palabras
    construye un VOCABULARIO con todas las palabras distintas de los mensajes de
    entrenamiento, cada una con su índice, y representa cada mensaje por el vector x con el
    número de veces que aparece en él cada palabra del vocabulario: x[i] es el número de
    veces que aparece la palabra de índice i.

        "win money, win"  -->  [..., 1, ..., 2, ...]
                                     ^money  ^win

    Se pierde el orden de las palabras (de ahí lo de «bolsa»), pero se conserva cuáles
    aparecen y cuántas veces.

    Hace lo mismo que CountVectorizer de scikit-learn, y parte el texto en palabras de la
    misma forma (en minúsculas, y solo las palabras de dos o más letras o cifras), así que
    la matriz que produce debe ser IDÉNTICA a la de la librería.
"""
import re
import numpy as np

# La expresión regular de CountVectorizer: palabras de 2 o más caracteres alfanuméricos.
# Las palabras de una sola letra ("a", "I") se descartan.
PATRON_PALABRA = re.compile(r'(?u)\b\w\w+\b')


def tokenizar(mensaje):
    """
    Parte un mensaje en la lista de sus palabras, en minúsculas y sin los signos de
    puntuación: 'I win, you WIN' -> ['win', 'you', 'win'].
    """
    return PATRON_PALABRA.findall(mensaje.lower())


def construir_vocabulario(mensajes):
    """
    Construye el vocabulario con las palabras de los mensajes de ENTRENAMIENTO: un
    diccionario {palabra: índice}, con las palabras distintas por orden alfabético (la
    primera tiene el índice 0, la segunda el 1...).
    """
    vocabulario = {}
    palabras = set()
    for mensaje in mensajes:
        palabras.update(tokenizar(mensaje))
    for i, palabra in enumerate(sorted(palabras)):  # el orden decide los índices
        vocabulario[palabra] = i
    return vocabulario


def vectorizar(mensajes, vocabulario):
    """
    Convierte una lista de mensajes en su matriz de conteos: una fila por mensaje y una
    columna por palabra del vocabulario, (n_mensajes, tamaño del vocabulario).
    """
    X = np.zeros((len(mensajes), len(vocabulario)), dtype=np.int32)
    for k, mensaje in enumerate(mensajes):
        for palabra in tokenizar(mensaje):
            if palabra in vocabulario:          # si no está en el vocabulario, se ignora
                X[k, vocabulario[palabra]] += 1
    return X


def palabras_del_vocabulario(vocabulario):
    """
    Las palabras del vocabulario en una lista, cada una en la posición de su índice.
    """
    palabras = [''] * len(vocabulario)
    for palabra, i in vocabulario.items():
        palabras[i] = palabra
    return palabras
