"""
    EJERCICIO 3 d): INTENTE ENGAÑAR AL FILTRO DE SPAM.

    Entrena el filtro de los ejercicios 2 y 3 con los 4.180 SMS de entrenamiento y
    clasifica los mensajes que usted escriba. De cada uno escribe si es spam, con
    P(spam|mensaje), el peso de cada una de sus palabras y las que el filtro no conoce,
    que no cuentan. El peso de una palabra es log P(palabra|spam) - log P(palabra|ham):
    positivo si empuja hacia spam y negativo si empuja hacia ham (y si sale dos veces,
    cuenta dos veces).

        python filtro.py                         pide mensajes hasta una línea vacía
        python filtro.py "mensaje" "otro..."     clasifica los mensajes que se le pasan
"""
import sys
import numpy as np
from naive_bayes.bolsa_de_palabras import construir_vocabulario, vectorizar, tokenizar
from naive_bayes.multinomial import (entrenar_naive_bayes_multinomial,
                                     calcular_puntuaciones_multinomial)
from naive_bayes.datos import cargar_sms, train_test_split


def entrenar_filtro():
    mensajes, y = cargar_sms()
    textos_train, textos_test, y_train, y_test = train_test_split(mensajes, y,
                                                                  test_size=0.25, seed=0)
    vocabulario = construir_vocabulario(textos_train)
    X_train = vectorizar(textos_train, vocabulario)
    log_priors, log_prob_palabras = entrenar_naive_bayes_multinomial(X_train, y_train, 2,
                                                                     alpha=1.0)
    return vocabulario, log_priors, log_prob_palabras


def clasificar(vocabulario, log_priors, log_prob_palabras, mensaje):
    x = vectorizar([mensaje], vocabulario)[0]
    puntuacion = calcular_puntuaciones_multinomial(x, log_priors, log_prob_palabras)
    # las puntuaciones son log P(c) + log P(mensaje|c): se deshace el logaritmo y se divide
    # por la suma de las dos, para que P(ham|mensaje) + P(spam|mensaje) = 1
    probabilidades = np.exp(puntuacion - puntuacion.max())
    probabilidades = probabilidades / probabilidades.sum()
    print('  -> %s   P(spam|mensaje) = %.4f'
          % ('SPAM' if puntuacion[1] > puntuacion[0] else 'HAM', probabilidades[1]))
    peso = log_prob_palabras[1] - log_prob_palabras[0]
    palabras = tokenizar(mensaje)
    conocidas = sorted(set(p for p in palabras if p in vocabulario),
                       key=lambda p: -peso[vocabulario[p]])
    desconocidas = [p for p in palabras if p not in vocabulario]
    print('     peso de cada palabra: %s' % (', '.join('%s %+.2f' % (p, peso[vocabulario[p]])
                                                     for p in conocidas) or 'ninguna'))
    print('     palabras que no conoce (no cuentan): %s' % (', '.join(desconocidas) or 'ninguna'))


def main(mensajes):
    print('Entrenando el filtro con los 4.180 SMS de entrenamiento...')
    vocabulario, log_priors, log_prob_palabras = entrenar_filtro()
    for mensaje in mensajes:
        print('\n%s' % mensaje)
        clasificar(vocabulario, log_priors, log_prob_palabras, mensaje)
    if mensajes:
        return
    while True:
        try:
            mensaje = input('\nMensaje (Intro para terminar): ')
        except EOFError:
            break
        if mensaje.strip() == '':
            break
        clasificar(vocabulario, log_priors, log_prob_palabras, mensaje)


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except NotImplementedError as error:
        print('El filtro todavía no funciona (%s): complete antes los ejercicios 2 y 3.' % error)
        raise SystemExit(1)
