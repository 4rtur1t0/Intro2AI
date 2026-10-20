"""
    EL TRES EN RAYA, ESCRITO COMO UNA CLASE.

    Se da completo. Los cuatro componentes de la definición formal de un juego, y el
    turno, son métodos de la clase:

        start_state()     1 · estado inicial: el tablero vacío, y le toca a X
        player(s)         a quién le toca: "MAX" (X) o "MIN" (O)
        successors(s)     2 · función sucesor: lista de pares (acción, siguiente estado)
        is_end(s)         3 · test terminal: ¿ha terminado la partida?
        utility(s)        4 · función utilidad: +1 gana X, -1 gana O, 0 tablas
                              (SIEMPRE desde el punto de vista de MAX)

    Un estado es la pareja (tablero, turno). El tablero es una lista de 9 casillas,
    numeradas así:

             0 | 1 | 2        'X'  juega MAX (y empieza)
            ---+---+---       'O'  juega MIN
             3 | 4 | 5        ' '  casilla libre
            ---+---+---
             6 | 7 | 8

    Son SOLO LAS REGLAS: ningún método sabe nada de estrategia. Los algoritmos
    (Minimax, AlfaBeta, MCTS) reciben un objeto game y llaman a estos métodos sin saber
    a qué se juega; por eso su MCTS puede jugar también a Conecta 4 sin cambiar nada.

    Además de las reglas, el módulo trae:
        TresEnRayaOrdenado    el mismo juego con otro orden de sucesores (ejercicio 2)
        estado_desde_texto, clave, todas_las_posiciones y dibujar: utilidades.
"""
# las 8 líneas ganadoras: tres filas, tres columnas y dos diagonales
LINEAS = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
          (0, 3, 6), (1, 4, 7), (2, 5, 8),
          (0, 4, 8), (2, 4, 6)]


def ganador(tablero):
    """
    Devuelve 'X' u 'O' si ese jugador tiene tres fichas en alguna línea, y None si no.
    """
    for a, b, c in LINEAS:
        if tablero[a] != ' ' and tablero[a] == tablero[b] == tablero[c]:
            return tablero[a]
    return None


class TresEnRaya:
    # 1 · Estado inicial: tablero vacío, y le toca a X
    def start_state(self):
        tablero = [' '] * 9
        return (tablero, 'X')

    # El turno va en el estado: X es MAX y O es MIN
    def player(self, state):
        tablero, turno = state
        if turno == 'X':
            return "MAX"
        return "MIN"

    # 2 · Función sucesor: par_salida (acción, siguiente estado)
    def successors(self, state):
        tablero, turno = state
        rival = 'O' if turno == 'X' else 'X'
        par_salida = []
        for casilla in range(9):
            if tablero[casilla] == ' ':  # acción legal
                nuevo_tablero = list(tablero)    # una COPIA del tablero
                nuevo_tablero[casilla] = turno
                par_salida.append((casilla, (nuevo_tablero, rival)))
        return par_salida

    # 3 · Test terminal: tres en raya, o tablero lleno
    def is_end(self, state):
        tablero, turno = state
        return ganador(tablero) is not None or ' ' not in tablero

    # 4 · Función utilidad: siempre desde el punto de vista de MAX
    def utility(self, state):
        tablero, turno = state
        puntos = {'X': +1, 'O': -1, None: 0}    # gana X, O o nadie
        return puntos[ganador(tablero)]


class TresEnRayaOrdenado(TresEnRaya):
    """
    El mismo juego con los sucesores en OTRO ORDEN (ejercicio 2): primero el centro,
    que está en 4 líneas, después las esquinas (3 líneas) y por último los lados (2).
    Con al_reves=True, justo al revés. Las reglas no cambian: solo el orden en el que
    los algoritmos miran las jugadas.
    """
    PRIORIDAD = {4: 0, 0: 1, 2: 1, 6: 1, 8: 1, 1: 2, 3: 2, 5: 2, 7: 2}

    def __init__(self, al_reves=False):
        self.al_reves = al_reves

    def successors(self, state):
        pares = sorted(TresEnRaya.successors(self, state),
                       key=lambda par: self.PRIORIDAD[par[0]])
        if self.al_reves:
            pares.reverse()
        return pares


# ----------------------------------------------------------------------------------
# Utilidades de la práctica (no forman parte del juego)
# ----------------------------------------------------------------------------------
def estado_desde_texto(texto):
    """
    Construye un estado a partir de 9 caracteres leídos fila a fila: 'X', 'O' o '.'
    para una casilla libre. El turno se deduce contando fichas: como empieza X, le toca
    a X si hay tantas X como O. Por ejemplo, estado_desde_texto('.X..OXXOO').
    """
    tablero = [' ' if c == '.' else c for c in texto]
    turno = 'X' if tablero.count('X') == tablero.count('O') else 'O'
    return (tablero, turno)


def clave(state):
    """
    El estado convertido en algo que puede ser clave de un diccionario: una lista no
    puede serlo, una tupla sí.
    """
    tablero, turno = state
    return (tuple(tablero), turno)


def todas_las_posiciones(game):
    """
    Todas las posiciones alcanzables desde el tablero vacío, sin repetir, en un
    diccionario {clave: estado}. Se recorren con la lista abierta y los visitados de
    la búsqueda en grafo de la práctica 2.1.
    """
    visitados = {}
    abiertos = [game.start_state()]
    while abiertos:
        state = abiertos.pop()
        if clave(state) in visitados:
            continue
        visitados[clave(state)] = state
        if not game.is_end(state):
            for accion, sucesor in game.successors(state):
                abiertos.append(sucesor)
    return visitados


def dibujar(state, valores=None):
    """
    Escribe el tablero en el terminal. En cada casilla libre pone su número o, si se
    pasa `valores` (un diccionario {casilla: valor}), ese valor: por ejemplo, el valor
    minimax de jugar ahí (+1 gana X, 0 tablas, -1 gana O).
    """
    tablero, turno = state
    textos = []
    for casilla in range(9):
        if tablero[casilla] != ' ':
            textos.append(tablero[casilla])
        elif valores is not None and casilla in valores:
            textos.append(_como_texto(valores[casilla]))
        else:
            textos.append(str(casilla))
    ancho = max(len(t) for t in textos) + 2
    for fila in range(3):
        print('|'.join(textos[3 * fila + col].center(ancho) for col in range(3)))
        if fila < 2:
            print('+'.join('-' * ancho for _ in range(3)))
    print()


def _como_texto(valor):
    if isinstance(valor, str):
        return valor
    if isinstance(valor, int):
        return '%+d' % valor if valor != 0 else '0'
    return '%+.2f' % valor
