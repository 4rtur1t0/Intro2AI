"""
    HERRAMIENTAS DE LA PRÁCTICA. Se dan completas: no hay que modificar nada.

    - comprobar(...): compara lo obtenido con lo esperado y escribe [BIEN] o [FALLA]
      (con una pista si falla). resumen() cuenta cuántas han salido bien.
    - intentar(...): ejecuta un apartado de un programa; si falta algún ejercicio por
      completar (NotImplementedError) o salta un error, lo dice y sigue con el siguiente.
    - construir_arbol(...) y dibujar_arbol(...): construyen un árbol de MCTS a partir de
      sus contadores y lo escriben en el terminal (ejercicio 3).
    - Humano: el jugador que pide la jugada por teclado, en jugar.py. Cualquier objeto
      con un método decidir(state) es un jugador: Minimax, AlfaBeta y MCTS lo son.
    - figura_nodos(...): la figura del ejercicio 2, que se guarda en figures/.
"""
import os
import traceback

AZUL, VERDE, ROJO, GRIS = '#1d4e89', '#1c7a4d', '#b4451f', '#7a8699'
RESULTADOS = {'bien': 0, 'falla': 0}


def miles(n):
    """549946 -> '549.946', para leer los números grandes de un vistazo."""
    return '{:,}'.format(n).replace(',', '.')


# ----------------------------------------------------------------------------------
# Comprobaciones
# ----------------------------------------------------------------------------------
def comprobar(descripcion, obtenido, esperado, pista=''):
    """Escribe [BIEN] si obtenido == esperado (con tolerancia si son números reales)."""
    if isinstance(esperado, float) and isinstance(obtenido, (int, float)):
        bien = abs(obtenido - esperado) < 1e-4
    else:
        bien = obtenido == esperado
    if bien:
        RESULTADOS['bien'] += 1
        print('  [BIEN]   %s' % descripcion)
    else:
        RESULTADOS['falla'] += 1
        print('  [FALLA]  %s' % descripcion)
        print('           esperaba : %s' % (esperado,))
        print('           ha salido: %s' % (obtenido,))
        if pista:
            print('           PISTA: %s' % pista)
    return bien


def intentar(titulo, funcion):
    """Ejecuta un apartado. Si falta un ejercicio o salta un error, lo dice y sigue."""
    print('\n' + titulo)
    print('-' * len(titulo))
    try:
        funcion()
    except NotImplementedError as error:
        RESULTADOS['falla'] += 1
        print('  [FALTA]  %s' % error)
    except Exception:
        RESULTADOS['falla'] += 1
        print('  [ERROR]  el apartado se ha interrumpido por este error:')
        traceback.print_exc()


def resumen():
    print('\nComprobaciones: %d bien, %d mal o sin hacer.'
          % (RESULTADOS['bien'], RESULTADOS['falla']))


# ----------------------------------------------------------------------------------
# El árbol de MCTS
# ----------------------------------------------------------------------------------
def construir_arbol(game, state, arbol):
    """
    Construye con la clase Nodo de algoritmos/mcts.py el árbol descrito en arbol, un
    diccionario {nombre del nodo: (N, U)}, tal y como lo habría dejado MCTS desde el
    estado state. El nombre de un nodo son las jugadas que llevan hasta él desde la
    raíz, como en la figura del guion: 'X0 O3' es X en la casilla 0 y después O en la 3,
    y la raíz es ''. Como en expandir, los hijos de cada nodo salen de sus jugadas sin
    probar con pop(), de la última a la primera, y las demás se quedan sin probar.
    Devuelve un diccionario {nombre del nodo: nodo}.
    """
    from algoritmos.mcts import Nodo
    nodos = {'': Nodo(game, state)}
    for nombre in sorted(arbol, key=lambda k: len(k.split())):   # padres antes que hijos
        nodo = nodos[nombre]
        nodo.N, nodo.U = arbol[nombre]
        hijos = [k for k in arbol if len(k.split()) == len(nombre.split()) + 1
                 and k.split()[:-1] == nombre.split()]
        for k in sorted(hijos, key=_jugada, reverse=True):       # en el orden de pop()
            hijo = Nodo(game, dict(nodo.sin_probar)[_jugada(k)], nodo, _jugada(k))
            nodo.hijos.append(hijo)
            nodos[k] = hijo
        probadas = [_jugada(k) for k in hijos]
        nodo.sin_probar = [(a, s) for a, s in nodo.sin_probar if a not in probadas]
    return nodos


def _jugada(nombre):
    """'X0 O3' -> 3: la casilla de la última jugada del nombre."""
    return int(nombre.split()[-1][1:])


def nombre_del_nodo(nodo, game):
    """Lo contrario: el nombre de un nodo, como 'X0 O2'. El de la raíz es ''."""
    jugadas = []
    try:
        while nodo.padre is not None:
            jugadas.append(_ficha(game, nodo.padre.state) + str(nodo.accion))
            nodo = nodo.padre
    except AttributeError:                    # no es un nodo, o no está bien construido
        if nodo is None:
            return 'None'
        return 'un objeto %s mal construido' % type(nodo).__name__
    return ' '.join(reversed(jugadas))


def contadores(nodo):
    """Los contadores de un nodo, como en la figura: U/N ('1/2', '0.5/1'); la raíz, N."""
    if nodo.padre is None:
        return 'N = %d' % nodo.N
    return '%g/%d' % (nodo.U, nodo.N)


def dibujar_arbol(raiz, game, ucb1=None):
    """
    Escribe en el terminal el árbol de MCTS que cuelga de raiz, con los nombres de la
    figura del guion: X0 es X en la casilla 0. En cada nodo, U/N: las victorias de
    quien eligió la jugada entre los playouts que han pasado por ella. Si se pasa ucb1
    (por ejemplo, mcts.ucb1), también su UCB1. Al final de la línea, las jugadas que el
    nodo todavía no ha probado, o si es terminal.
    """
    print('raíz (le toca a %s)   %s%s' % (_ficha(game, raiz.state), contadores(raiz),
                                          _al_final(raiz, game)))
    _dibujar_hijos(raiz, game, ucb1, '')
    print()


def _dibujar_hijos(nodo, game, ucb1, prefijo):
    quien = _ficha(game, nodo.state)          # el que elige entre los hijos
    hijos = sorted(nodo.hijos, key=lambda h: h.accion)
    for k, hijo in enumerate(hijos):
        ultimo = k == len(hijos) - 1
        texto = '%s%s   U/N = %s' % (quien, hijo.accion, contadores(hijo))
        if ucb1 is not None:                  # con N = 0, UCB1 no está definida
            texto += '   UCB1 = ' + ('%.4f' % ucb1(hijo) if hijo.N > 0 else '?')
        print(prefijo + ('└─ ' if ultimo else '├─ ') + texto + _al_final(hijo, game))
        _dibujar_hijos(hijo, game, ucb1, prefijo + ('   ' if ultimo else '│  '))


def _ficha(game, state):
    return 'X' if game.player(state) == "MAX" else 'O'


def _al_final(nodo, game):
    if game.is_end(nodo.state):
        u = game.utility(nodo.state)
        return '   (terminal: %s)' % ('tablas' if u == 0 else 'gana X' if u > 0 else 'gana O')
    if nodo.sin_probar:
        quien = _ficha(game, nodo.state)
        return '   (sin probar: %s)' % ', '.join(quien + str(a) for a, s in nodo.sin_probar)
    return ''


# ----------------------------------------------------------------------------------
# Jugadores
# ----------------------------------------------------------------------------------
class Humano:
    """Pide la jugada por teclado."""
    def __init__(self, game, dibujar):
        self.game = game
        self.dibujar = dibujar

    def decidir(self, state):
        self.dibujar(state)
        legales = [accion for accion, sucesor in self.game.successors(state)]
        while True:
            texto = input('Su jugada (%s): ' % ', '.join(str(a) for a in legales))
            try:
                accion = int(texto.strip())
            except ValueError:
                continue
            if accion in legales:
                return accion
            print('Esa jugada no es legal.')


def nombre_de(jugador):
    """Un nombre corto para escribir resultados: 'MCTS 1000', 'alfa-beta'..."""
    clase = type(jugador).__name__
    if clase == 'MCTS':
        return 'MCTS %d' % jugador.iteraciones
    if clase == 'AlfaBeta':
        return 'alfa-beta'
    if clase == 'Minimax':
        return 'minimax'
    return clase


# ----------------------------------------------------------------------------------
# La figura del ejercicio 2
# ----------------------------------------------------------------------------------
def _guardar(fig, ruta):
    import matplotlib
    import matplotlib.pyplot as plt
    fig.tight_layout()
    # la ruta es relativa al directorio de la práctica, se ejecute desde donde se ejecute
    completa = os.path.join(os.path.dirname(os.path.abspath(__file__)), ruta)
    os.makedirs(os.path.dirname(completa), exist_ok=True)
    fig.savefig(completa, dpi=150)
    print('   figura guardada en %s' % ruta)
    if matplotlib.get_backend().lower() != 'agg':
        plt.show()
    plt.close(fig)


def figura_nodos(cuentas, ruta):
    """Ejercicio 2: nodos visitados por cada algoritmo, en escala logarítmica."""
    import math
    import matplotlib.pyplot as plt
    etiquetas, nodos = list(cuentas.keys()), list(cuentas.values())
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    y = list(range(len(nodos)))[::-1]
    ax.barh(y, nodos, color=[GRIS, AZUL, ROJO, VERDE][:len(nodos)])
    for yi, n in zip(y, nodos):
        ax.text(n * 1.1, yi, miles(n), va='center', fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(etiquetas)
    raiz = math.sqrt(max(nodos))
    ax.axvline(raiz, color=VERDE, ls=':', lw=1.3,
               label='raíz cuadrada del árbol completo (%d)' % round(raiz))
    ax.set_xscale('log')
    ax.set_xlim(right=max(nodos) * 5)
    ax.set_xlabel('nodos visitados desde el tablero vacío (escala logarítmica)')
    ax.set_title('El mismo valor (0) en todas las filas: solo cambia el trabajo')
    ax.legend(fontsize=8, loc='lower right')
    _guardar(fig, ruta)
