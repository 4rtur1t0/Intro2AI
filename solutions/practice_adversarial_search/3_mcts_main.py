"""
    PRÁCTICA 2.2. BÚSQUEDA CON ADVERSARIO. EJERCICIO 3: MCTS, PASO A PASO.

    Complete algoritmos/mcts.py en el orden de los apartados y ejecute este programa
    después de cada uno: cada apartado comprueba SOLO su paso. Los cuatro siguen la
    décima iteración de MCTS en la posición '.X..OXXOO' (figura 1c del guion), sobre el
    árbol que han construido las nueve primeras (figura 4), que aquí es ARBOL.

        a) ucb1 y seleccionar   escribe el árbol con la UCB1 de cada nodo y comprueba
                                dónde se para seleccionar(raíz)
        b) expandir             expande el nodo en el que se para la selección
        c) simular              simula una vez desde el estado del nodo creado en b) y
                                10.000 veces desde el tablero vacío
        d) retropropagar        retropropaga desde el nodo creado en b) un +1 (gana X),
                                un 0 (tablas) y un -1 (gana O), cada uno en un árbol nuevo

    Ejecútelo desde este directorio:   python 3_mcts_main.py
"""
import random
from juegos.tres_en_raya import TresEnRaya, estado_desde_texto, dibujar
from algoritmos.mcts import MCTS
from herramientas import (comprobar, intentar, resumen, construir_arbol, nombre_del_nodo,
                          contadores, dibujar_arbol)

POSICION = '.X..OXXOO'
# El árbol de la figura 4. Para cada nodo, su nombre (las jugadas que llevan hasta él
# desde la raíz: 'X0 O2' es X en 0 y después O en 2) y sus contadores (N, U).
ARBOL = {'':         (9, 0),
         'X0':       (4, 4),
         'X0 O2':    (1, 0),
         'X0 O3':    (2, 0),
         'X0 O3 X2': (1, 1),
         'X2':       (2, 1),
         'X2 O3':    (1, 0),
         'X3':       (3, 2),
         'X3 O0':    (1, 1),
         'X3 O2':    (1, 0)}


def seleccion(game):
    nodos = construir_arbol(game, estado_desde_texto(POSICION), ARBOL)
    mcts = MCTS(game, 0)                       # C = 1,4
    valores = [round(mcts.ucb1(nodos[k]), 4) for k in ('X0', 'X2', 'X3')]
    print('El árbol de la figura 4, con la UCB1 de cada nodo:')
    dibujar_arbol(nodos[''], game, mcts.ucb1)
    comprobar('UCB1 de X0, X2 y X3', valores, [2.0376, 1.9674, 1.8648],
              'U/N + C * sqrt(log(N del padre) / N), con el logaritmo neperiano.')
    nodo = mcts.seleccionar(nodos[''])
    print('\nseleccionar(raíz) se para en el nodo %s.' % nombre_del_nodo(nodo, game))
    comprobar('seleccionar(raíz) baja por X0 y por O2, y se para ahí',
              nombre_del_nodo(nodo, game), 'X0 O2',
              'Hay que bajar al hijo de mayor UCB1 MIENTRAS el nodo no tenga jugadas sin '
              'probar y sí tenga hijos: un bucle, no un solo paso.')
    terminal = nodos['X0 O3 X2']
    comprobar('seleccionar(X0 O3 X2), un nodo terminal, se queda en él',
              mcts.seleccionar(terminal) is terminal, True,
              'Un nodo terminal no tiene hijos: no hay por dónde bajar.')


def expansion(game):
    nodos = construir_arbol(game, estado_desde_texto(POSICION), ARBOL)
    nodo = nodos['X0 O2']                      # donde se para la selección del a)
    hijo = MCTS(game, 0).expandir(nodo)
    print('expandir(X0 O2) devuelve el nodo %s.' % nombre_del_nodo(hijo, game))
    if not comprobar('expandir(X0 O2) crea el nodo X0 O2 X3 y lo devuelve',
                     nombre_del_nodo(hijo, game), 'X0 O2 X3',
                     'Saque de sin_probar la última pareja (acción, sucesor) y cree con ella '
                     'el hijo: mire el orden de los argumentos del constructor de Nodo.'):
        return
    comprobar('X0 O2 X3 está entre los hijos de X0 O2, y a X0 O2 ya no le quedan jugadas '
              'sin probar', (hijo in nodo.hijos, nodo.sin_probar), (True, []),
              'pop() quita la pareja de sin_probar, y append añade el hijo a nodo.hijos.')
    comprobar('el estado de X0 O2 X3: X ha jugado en la casilla 3', hijo.state,
              estado_desde_texto('XXOXOXXOO'),
              'El estado del hijo es el sucesor que acompaña a la jugada en sin_probar.')
    print('\nEl árbol, con el nodo nuevo:')
    dibujar_arbol(nodos[''], game)
    terminal = nodos['X0 O3 X2']
    comprobar('expandir(X0 O3 X2), un nodo terminal, lo devuelve tal cual',
              MCTS(game, 0).expandir(terminal) is terminal, True,
              'Un nodo terminal no tiene jugadas sin probar: no hay nada que expandir.')


def simulacion(game):
    mcts = MCTS(game, 0)
    s = estado_desde_texto('XXOXOXXOO')        # el estado del nodo creado en b)
    print('El estado del nodo X0 O2 X3, creado en b):')
    dibujar(s)
    resultado = mcts.simular(s)
    print('simular(ese estado) devuelve %s.' % (resultado,))
    comprobar('simular desde el estado de X0 O2 X3 devuelve +1', resultado, 1,
              'Si la partida ya ha terminado, no hay nada que jugar: devuelva su utilidad.')

    random.seed(0)                             # siempre el mismo azar: los números del guion
    resultados = [mcts.simular(game.start_state()) for _ in range(10000)]
    porcentajes = [round(100 * resultados.count(r) / len(resultados), 1) for r in (+1, -1, 0)]
    print('\n10.000 partidas al azar desde el tablero vacío: gana X el %.1f %%, gana O el '
          '%.1f %% y hay tablas en el %.1f %%.' % tuple(porcentajes))
    comprobar('% de victorias de X, de O y de tablas', porcentajes, [58.6, 28.4, 13.0],
              'Use random.choice con la lista que da successors, una sola vez por jugada.')


def retropropagacion(game):
    s = estado_desde_texto(POSICION)
    arbol = dict(ARBOL)
    arbol['X0 O2 X3'] = (0, 0)                 # el árbol con el nodo nuevo del b)
    camino = ['X0 O2 X3', 'X0 O2', 'X0', '']   # del nodo nuevo a la raíz
    casos = [(+1, '+1 (gana X)', ['1/1', '0/2', '5/5', 'N = 10']),
             (0, '0 (tablas)', ['0.5/1', '0.5/2', '4.5/5', 'N = 10']),
             (-1, '-1 (gana O)', ['0/1', '1/2', '4/5', 'N = 10'])]
    print('Desde el nodo nuevo, X0 O2 X3, hasta la raíz: U/N de X3, O2 y X0, y N de la raíz.')
    for resultado, texto, esperado in casos:
        nodos = construir_arbol(game, s, arbol)        # cada resultado, en un árbol nuevo
        MCTS(game, 0).retropropagar(nodos['X0 O2 X3'], resultado)
        comprobar('resultado %s: %s' % (texto, ', '.join(esperado)),
                  [contadores(nodos[k]) for k in camino], esperado,
                  'Cada nodo del camino suma 1 a N y, si tiene padre, a U la victoria de quien '
                  'lo eligió: el jugador que movía en el PADRE. Si es MIN, cambie el signo '
                  'del resultado.')
        if resultado == +1:
            raiz = nodos['']
    print('\nEl árbol, después de retropropagar el +1:')
    dibujar_arbol(raiz, game)


if __name__ == "__main__":
    game = TresEnRaya()
    intentar('a) SELECCIÓN: ucb1 y seleccionar', lambda: seleccion(game))
    intentar('b) EXPANSIÓN: expandir', lambda: expansion(game))
    intentar('c) SIMULACIÓN: simular', lambda: simulacion(game))
    intentar('d) RETROPROPAGACIÓN: retropropagar', lambda: retropropagacion(game))
    resumen()
