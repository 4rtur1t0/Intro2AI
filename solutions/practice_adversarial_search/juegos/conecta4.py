"""
    CONECTA 4, CON LA MISMA INTERFAZ QUE EL TRES EN RAYA (OPCIONAL).

    Se da completo. Tiene los mismos métodos que TresEnRaya (start_state, player,
    successors, is_end, utility), así que su MCTS juega a Conecta 4 sin cambiar ni una
    línea:   python jugar.py --juego conecta4

    Minimax y alfa-beta, en cambio, tendrían que llegar al final de la partida, y el
    árbol de Conecta 4 tiene del orden de 10^21 nodos.

    El tablero tiene 6 filas y 7 columnas y se guarda en una lista de 42 casillas: la
    casilla (fila, columna) es la posición 7 * fila + columna, y la fila 0 es la de
    ARRIBA. Una acción es una COLUMNA: la ficha cae hasta la casilla libre más baja.

    El estado es la terna (tablero, turno, última casilla ocupada). Guardar la última
    casilla permite saber si alguien acaba de ganar mirando solo las líneas que pasan
    por la ficha recién puesta, en lugar de recorrer el tablero entero: con miles de
    simulaciones por jugada, la diferencia se nota.
"""
FILAS, COLUMNAS = 6, 7


def ventanas_de_cuatro():
    """Las 69 ventanas de 4 casillas seguidas: horizontales, verticales y diagonales."""
    ventanas = []
    for fila in range(FILAS):
        for columna in range(COLUMNAS):
            for df, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                casillas = [(fila + i * df, columna + i * dc) for i in range(4)]
                if all(0 <= f < FILAS and 0 <= c < COLUMNAS for f, c in casillas):
                    ventanas.append(tuple(COLUMNAS * f + c for f, c in casillas))
    return ventanas


VENTANAS = ventanas_de_cuatro()
# las ventanas que pasan por cada casilla
VENTANAS_DE = {casilla: [v for v in VENTANAS if casilla in v]
               for casilla in range(FILAS * COLUMNAS)}


class Conecta4:
    # 1 · Estado inicial: tablero vacío, le toca a X y aún no hay última casilla
    def start_state(self):
        tablero = [' '] * (FILAS * COLUMNAS)
        return (tablero, 'X', None)

    # El turno va en el estado: X es MAX y O es MIN
    def player(self, state):
        tablero, turno, ultima = state
        if turno == 'X':
            return "MAX"
        return "MIN"

    # 2 · Función sucesor: una acción por cada columna que no esté llena
    def successors(self, state):
        tablero, turno, ultima = state
        rival = 'O' if turno == 'X' else 'X'
        par_salida = []
        for columna in range(COLUMNAS):
            if tablero[columna] == ' ':          # la casilla de arriba, libre
                nuevo_tablero = list(tablero)    # una COPIA del tablero
                casilla = self.casilla_libre(tablero, columna)
                nuevo_tablero[casilla] = turno
                par_salida.append((columna, (nuevo_tablero, rival, casilla)))
        return par_salida

    def casilla_libre(self, tablero, columna):
        """La casilla libre más baja de la columna: ahí cae la ficha."""
        for fila in range(FILAS - 1, -1, -1):
            if tablero[COLUMNAS * fila + columna] == ' ':
                return COLUMNAS * fila + columna
        return None

    def ganador(self, state):
        """'X' u 'O' si la última ficha puesta ha hecho cuatro en raya; si no, None."""
        tablero, turno, ultima = state
        if ultima is None:
            return None
        ficha = tablero[ultima]
        for ventana in VENTANAS_DE[ultima]:
            if all(tablero[i] == ficha for i in ventana):
                return ficha
        return None

    # 3 · Test terminal: cuatro en raya, o tablero lleno (fila de arriba ocupada)
    def is_end(self, state):
        tablero, turno, ultima = state
        return self.ganador(state) is not None or ' ' not in tablero[:COLUMNAS]

    # 4 · Función utilidad: siempre desde el punto de vista de MAX
    def utility(self, state):
        puntos = {'X': +1, 'O': -1, None: 0}    # gana X, O o nadie
        return puntos[self.ganador(state)]


# ----------------------------------------------------------------------------------
# Utilidades de la práctica (no forman parte del juego)
# ----------------------------------------------------------------------------------
def dibujar(state, valores=None):
    """Escribe el tablero en el terminal, con los números de columna debajo."""
    tablero, turno, ultima = state
    for fila in range(FILAS):
        casillas = [tablero[COLUMNAS * fila + c] for c in range(COLUMNAS)]
        print('|' + '|'.join(c if c != ' ' else '.' for c in casillas) + '|')
    print(' ' + ' '.join(str(c) for c in range(COLUMNAS)))
    print()
