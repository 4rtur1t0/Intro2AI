"""
    PRÁCTICA 2.2. BÚSQUEDA CON ADVERSARIO. EJERCICIO 1: MINIMAX.

    Completar: algoritmos/minimax.py (Minimax.valor, valor_max y valor_min).

        a) valor minimax y nodos visitados en las dos posiciones de la figura 1 del guion
        b) lo mismo desde el tablero vacío, y el valor de cada apertura de X
        c) no está aquí: juegue contra su minimax con   python jugar.py --rival minimax

    Ejecútelo desde este directorio:   python 1_minimax_main.py
"""
import time
from juegos.tres_en_raya import TresEnRaya, estado_desde_texto, dibujar
from algoritmos.minimax import Minimax
from herramientas import comprobar, intentar, resumen, miles


def posiciones_pequenas(game):
    s = estado_desde_texto('.X..OXXOO')
    print("Posición '.X..OXXOO'; le toca a %s:" % s[1])
    dibujar(s)
    m = Minimax(game)
    v = m.valor(s)        # una sola llamada: cada llamada vuelve a sumar en m.nodos
    print('Minimax(game).valor(s) = %s   (%d nodos visitados)' % (v, m.nodos))
    comprobar("Minimax(game).valor(s) = (valor, jugada)", v, (1, 0),
              'X gana jugando en 0. Revise los tres casos de valor y que valor_max y '
              'valor_min devuelvan la pareja (mejor, jugada).')
    comprobar('nodos visitados', m.nodos, 14,
              'Cada sucesor se visita una vez, llamando a self.valor(sucesor).')

    s = estado_desde_texto('XX.OO.X..')
    print("\nPosición 'XX.OO.X..'; le toca a %s:" % s[1])
    dibujar(s)
    v = Minimax(game).valor(s)
    print('Minimax(game).valor(s) = %s' % (v,))
    comprobar("Minimax(game).valor(s) = (valor, jugada)", v, (-1, 5),
              'Mueve O, que es MIN: hay que quedarse con el MÍNIMO de los sucesores.')


def tablero_vacio(game):
    s0 = game.start_state()
    m = Minimax(game)
    inicio = time.perf_counter()
    v = m.valor(s0)
    print('Valor minimax del tablero vacío: %s   (%s nodos, %.2f s)'
          % (v, miles(m.nodos), time.perf_counter() - inicio))
    comprobar('(valor, jugada) del tablero vacío', v, (0, 0))
    comprobar('nodos visitados', m.nodos, 549946)

    print('\nValor minimax de cada apertura de X (+1 gana X, 0 tablas, -1 gana O):')
    valores = {accion: Minimax(game).valor(sucesor)[0]
               for accion, sucesor in game.successors(s0)}
    dibujar(s0, valores)
    print('decidir(tablero vacío) ->', Minimax(game).decidir(s0))


if __name__ == "__main__":
    game = TresEnRaya()
    intentar('a) LAS DOS POSICIONES DE LA FIGURA 1', lambda: posiciones_pequenas(game))
    intentar('b) EL TABLERO VACÍO', lambda: tablero_vacio(game))
    resumen()
