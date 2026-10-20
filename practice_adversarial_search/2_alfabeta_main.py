"""
    PRÁCTICA 2.2. BÚSQUEDA CON ADVERSARIO. EJERCICIO 2: PODA ALFA-BETA.

    Completar: algoritmos/alfabeta.py (AlfaBeta.valor, valor_max y valor_min).
    Necesita el ejercicio 1 hecho: compara alfa-beta con su Minimax.

        a) valor y nodos visitados en la posición de la figura 1c, con y sin poda
        b) lo mismo desde el tablero vacío, y en cuántas de las 5.478 posiciones del
           juego dan minimax y alfa-beta un resultado distinto
        c) nodos visitados con los sucesores en otro orden, y la figura de los nodos

    Ejecútelo desde este directorio:   python 2_alfabeta_main.py
"""
import math
import time
from juegos.tres_en_raya import (TresEnRaya, TresEnRayaOrdenado, estado_desde_texto,
                                 todas_las_posiciones)
from algoritmos.minimax import Minimax
from algoritmos.alfabeta import AlfaBeta
from herramientas import comprobar, intentar, resumen, miles, figura_nodos


def posicion_pequena(game):
    s = estado_desde_texto('.X..OXXOO')
    m, ab = Minimax(game), AlfaBeta(game)
    vm = m.valor(s)
    vab = ab.valor(s, -math.inf, +math.inf)
    print("Posición '.X..OXXOO': minimax %s con %d nodos; alfa-beta %s con %d nodos"
          % (vm, m.nodos, vab, ab.nodos))
    comprobar('alfa-beta: el mismo (valor, jugada) que minimax', vab, (1, 0))
    comprobar('alfa-beta: nodos visitados', ab.nodos, 10,
              'Con la poda deben ser menos que los 14 de minimax. ¿Está el break?')


def tablero_vacio(game, cuentas):
    s0 = game.start_state()
    m, ab = Minimax(game), AlfaBeta(game)
    inicio = time.perf_counter()
    vm = m.valor(s0)
    t_m = time.perf_counter() - inicio
    inicio = time.perf_counter()
    vab = ab.valor(s0, -math.inf, +math.inf)
    t_ab = time.perf_counter() - inicio
    print('minimax   : (valor, jugada) = %s   %9s nodos   %.2f s' % (vm, miles(m.nodos), t_m))
    print('alfa-beta : (valor, jugada) = %s   %9s nodos   %.2f s' % (vab, miles(ab.nodos), t_ab))
    print('alfa-beta visita %.1f veces menos nodos' % (m.nodos / ab.nodos))
    comprobar('alfa-beta: el mismo (valor, jugada) que minimax', vab, vm)
    comprobar('alfa-beta: nodos visitados', ab.nodos, 18297,
              'Si salen más, revise que alfa y beta se actualizan y se pasan al hijo; '
              'si salen menos, que la condición de poda es beta <= alfa.')
    cuentas['minimax'] = m.nodos
    cuentas['alfa-beta, orden natural'] = ab.nodos


def todas(game):
    posiciones = todas_las_posiciones(game)
    inicio = time.perf_counter()
    distintas = 0
    for s in posiciones.values():
        if Minimax(game).valor(s) != AlfaBeta(game).valor(s, -math.inf, +math.inf):
            distintas += 1
    print('Posiciones distintas del tres en raya: %s' % miles(len(posiciones)))
    print('En cuántas dan minimax y alfa-beta un valor o una jugada distintos: %d   (%.1f s)'
          % (distintas, time.perf_counter() - inicio))
    comprobar('número de posiciones', len(posiciones), 5478)
    comprobar('posiciones con un resultado distinto', distintas, 0)


def orden(cuentas):
    s0 = TresEnRaya().start_state()
    casos = [('lados primero', 'lados -> esquinas -> centro', True, (0, 7), 42094),
             ('centro primero', 'centro -> esquinas -> lados', False, (0, 4), 7275)]
    for corto, largo, al_reves, esperado, nodos in casos:
        ab = AlfaBeta(TresEnRayaOrdenado(al_reves))
        v = ab.valor(s0, -math.inf, +math.inf)
        print('alfa-beta, %s: (valor, jugada) = %s   %s nodos' % (largo, v, miles(ab.nodos)))
        comprobar('orden %s: (valor, jugada) y nodos' % largo, (v, ab.nodos), (esperado, nodos))
        cuentas['alfa-beta, ' + corto] = ab.nodos
    print('Raíz cuadrada del árbol completo: %.0f nodos' % math.sqrt(549946))
    figura_nodos(cuentas, ruta='figures/ej2_nodos.png')


if __name__ == "__main__":
    game = TresEnRaya()
    cuentas = {}
    intentar('a) LA POSICIÓN PEQUEÑA, CON PODA', lambda: posicion_pequena(game))
    intentar('b) EL TABLERO VACÍO', lambda: tablero_vacio(game, cuentas))
    intentar('b) TODAS LAS POSICIONES DEL JUEGO', lambda: todas(game))
    intentar('c) EL ORDEN DE EXPLORACIÓN', lambda: orden(cuentas))
    resumen()
