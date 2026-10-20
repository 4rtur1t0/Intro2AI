"""
    EJERCICIO 1 · MINIMAX.

    El valor minimax de un estado es el resultado de la partida si, desde él, los dos
    jugadores juegan perfecto:

                  | Utilidad(s)                            si s es terminal
        V(s)  =   | max { V(s') : s' en Sucesores(s) }     si s es un estado MAX
                  | min { V(s') : s' en Sucesores(s) }     si s es un estado MIN

    valor es la fórmula, caso a caso; valor_max y valor_min son el máximo y el mínimo
    escritos como un bucle. Los tres devuelven la PAREJA (valor, jugada): en la raíz lo
    que interesa es la jugada, y eso es lo que devuelve decidir.

    La clase no sabe a qué se juega: solo llama a los métodos del objeto game
    (player, successors, is_end, utility).
"""
import math


class Minimax:
    def __init__(self, game):
        self.game = game            # las reglas del juego
        self.nodos = 0              # (práctica) llamadas a valor: una por nodo

    # La decisión minimax: la jugada que lleva al mejor valor
    def decidir(self, state):
        v, jugada = self.valor(state)
        return jugada

    # V(s), caso a caso. Devuelve la pareja (valor, jugada)
    def valor(self, state):
        self.nodos += 1             # (práctica) se cuenta el nodo
        ############################################################################
        # EJERCICIO 1: SE DEBE COMPLETAR EL CÓDIGO POR EL ESTUDIANTE.
        # Los tres casos de la fórmula, en este orden:
        #   - si el estado es terminal (self.game.is_end), devuelva la pareja
        #     (self.game.utility(state), None): en un estado terminal no hay jugada;
        #   - si le toca a "MAX" (self.game.player), devuelva self.valor_max(state);
        #   - si no, devuelva self.valor_min(state).
        ############################################################################
        raise NotImplementedError('EJERCICIO 1 sin completar: Minimax.valor()')

    def valor_max(self, state):
        ############################################################################
        # EJERCICIO 1: SE DEBE COMPLETAR EL CÓDIGO POR EL ESTUDIANTE.
        # El MÁXIMO de la fórmula, escrito como un bucle:
        #   - empiece con mejor, jugada = -math.inf, None;
        #   - recorra los pares (accion, sucesor) de self.game.successors(state);
        #   - pida el valor de cada sucesor, UN NIVEL MÁS ABAJO (la llamada recursiva):
        #         v, _ = self.valor(sucesor)
        #     (valor devuelve una pareja; la jugada del hijo no hace falta: _);
        #   - si v es mayor que mejor, quédese con él y con su acción;
        #   - al terminar, devuelva la pareja (mejor, jugada).
        ############################################################################
        raise NotImplementedError('EJERCICIO 1 sin completar: Minimax.valor_max()')

    def valor_min(self, state):
        ############################################################################
        # EJERCICIO 1: SE DEBE COMPLETAR EL CÓDIGO POR EL ESTUDIANTE.
        # El MÍNIMO de la fórmula: es valor_max con solo dos cambios.
        ############################################################################
        raise NotImplementedError('EJERCICIO 1 sin completar: Minimax.valor_min()')
