"""
    EJERCICIO 2 · PODA ALFA-BETA.

    AlfaBeta es la clase Minimax del ejercicio 1 con DOS COTAS Y UN CORTE:

        alfa : el mejor valor que MAX tiene ya asegurado en el camino de la raíz al
               nodo actual (cota inferior). Empieza en -infinito y solo sube.
        beta : el mejor valor que MIN tiene ya asegurado en ese camino (cota
               superior). Empieza en +infinito y solo baja.

    Después de cada hijo, MAX sube alfa y MIN baja beta; si beta <= alfa, la ventana
    se ha cerrado y el resto de hermanos sobra: break. Alfa y beta son PARÁMETROS de
    la llamada recursiva: lo que un hijo cambie en su copia no sube al padre ni pasa a
    sus hermanos. El resultado es EXACTAMENTE el mismo valor que el de minimax.
"""
import math


class AlfaBeta:
    def __init__(self, game):
        self.game = game            # las reglas del juego
        self.nodos = 0              # (práctica) llamadas a valor: una por nodo

    def decidir(self, state):
        # al empezar no hay nada asegurado
        v, jugada = self.valor(state, -math.inf, +math.inf)
        return jugada

    def valor(self, state, alfa, beta):
        self.nodos += 1             # (práctica) se cuenta el nodo
        ############################################################################
        # EJERCICIO 2: SE DEBE COMPLETAR EL CÓDIGO POR EL ESTUDIANTE.
        # El mismo valor que en Minimax, pasando alfa y beta a valor_max y valor_min.
        ############################################################################
        raise NotImplementedError('EJERCICIO 2 sin completar: AlfaBeta.valor()')

    def valor_max(self, state, alfa, beta):
        ############################################################################
        # EJERCICIO 2: SE DEBE COMPLETAR EL CÓDIGO POR EL ESTUDIANTE.
        # Copie su valor_max de Minimax y haga dos cambios:
        #   1) pida el valor de cada sucesor con self.valor(sucesor, alfa, beta);
        #   2) después de cada hijo, suba alfa y mire si la ventana se ha cerrado:
        #          alfa = max(alfa, mejor)
        #          if beta <= alfa:
        #              break                    # PODA: el resto sobra
        ############################################################################
        raise NotImplementedError('EJERCICIO 2 sin completar: AlfaBeta.valor_max()')

    def valor_min(self, state, alfa, beta):
        ############################################################################
        # EJERCICIO 2: SE DEBE COMPLETAR EL CÓDIGO POR EL ESTUDIANTE.
        # Lo simétrico en MIN: después de cada hijo se BAJA beta con
        # beta = min(beta, mejor), y se poda igual, si beta <= alfa.
        ############################################################################
        raise NotImplementedError('EJERCICIO 2 sin completar: AlfaBeta.valor_min()')
