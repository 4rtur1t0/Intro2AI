"""
    EJERCICIO 3 · MONTE CARLO TREE SEARCH (MCTS) CON UCT.

    MCTS no recorre el árbol entero ni usa una función de evaluación: construye un
    árbol parcial, un nodo por iteración, y estima el valor de cada jugada jugando
    partidas AL AZAR hasta el final. Cada iteración tiene cuatro pasos, y cada paso es
    un método que se completa en un apartado del ejercicio:

        1 · SELECCIÓN          a)  bajar por el árbol ya construido, siempre por el
                                   hijo de mayor UCB1, hasta un nodo con jugadas sin
                                   probar (o terminal).
        2 · EXPANSIÓN          b)  colgar de ese nodo UN hijo nuevo, con una jugada
                                   sin probar.
        3 · SIMULACIÓN         c)  desde el hijo nuevo, una partida al azar hasta el
                                   final.
        4 · RETROPROPAGACIÓN   d)  subir el resultado hasta la raíz, actualizando N y U.

    El bucle (decidir) y la clase Nodo ya están escritos: léalos antes de empezar.
    Los dos contadores de cada nodo son:
        N : playouts (simulaciones) que han pasado por el nodo.
        U : victorias de QUIEN LO ELIGIÓ, es decir, del jugador que movía en el nodo
            PADRE: gana 1, tablas 1/2, pierde 0.

    La política de selección UCT ordena los hijos con la fórmula UCB1:

        UCB1(n) = U(n) / N(n)  +  C * sqrt( log N(Padre(n)) / N(n) )
                  -----------     ----------------------------------
                    explotar                   explorar

    Al terminar se juega la jugada de la raíz MÁS JUGADA (con más N), no la de mejor
    media: 65/100 es mejor apuesta que 2/3.
"""
import math
import random


class MCTS:
    def __init__(self, game, iteraciones, C=1.4):
        self.game        = game
        self.iteraciones = iteraciones
        self.C           = C        # explorar frente a explotar

    def decidir(self, state):
        raiz = Nodo(self.game, state)
        self.raiz = raiz            # (práctica) se guarda para poder mirar el árbol
        for _ in range(self.iteraciones):
            hoja      = self.seleccionar(raiz)       # 1
            hijo      = self.expandir(hoja)          # 2
            resultado = self.simular(hijo.state)     # 3
            self.retropropagar(hijo, resultado)      # 4
        # se juega la jugada MÁS JUGADA, no la de mejor media
        mejor = max(raiz.hijos, key=lambda h: h.N)
        return mejor.accion

    # 1 · SELECCIÓN: bajar por el árbol, siempre por el mejor UCB1
    def seleccionar(self, nodo):
        ############################################################################
        # EJERCICIO 3 a): SE DEBE COMPLETAR EL CÓDIGO POR EL ESTUDIANTE.
        # Baje desde nodo, cada vez al hijo de mayor UCB1, MIENTRAS el nodo actual no
        # tenga jugadas sin probar (nodo.sin_probar) y sí tenga hijos (nodo.hijos).
        # Devuelva el nodo en el que se para: el primero con jugadas sin probar, o
        # uno terminal.
        # AYUDA: max(lista, key=f) devuelve el elemento x de la lista con mayor f(x).
        #        A key se le pasa la función f, sin llamarla: sin paréntesis.
        ############################################################################
        while nodo.sin_probar == [] and nodo.hijos != []:
            nodo = max(nodo.hijos, key=self.ucb1)
        return nodo
      

    def ucb1(self, n):              # explotar + explorar
        ############################################################################
        # EJERCICIO 3 a): SE DEBE COMPLETAR EL CÓDIGO POR EL ESTUDIANTE.
        # Devuelva la UCB1 del nodo n, con sus contadores (n.U y n.N), los playouts
        # de su padre (n.padre) y la constante self.C.
        # AYUDA: math.log es el logaritmo neperiano y math.sqrt, la raíz cuadrada.
        ############################################################################
        
        return (n.U / n.N) + self.C * math.sqrt(math.log(n.padre.N) / n.N)

    # 2 · EXPANSIÓN: un hijo nuevo, con una jugada sin probar
    def expandir(self, nodo):
        ############################################################################
        # EJERCICIO 3 b): SE DEBE COMPLETAR EL CÓDIGO POR EL ESTUDIANTE.
        #   - Si al nodo no le quedan jugadas sin probar, es terminal: devuélvalo tal
        #     cual, sin crear nada.
        #   - Si no, saque de nodo.sin_probar la ÚLTIMA pareja (acción, sucesor), cree
        #     el nodo hijo (mire el constructor de Nodo, al final del fichero),
        #     añádalo a los hijos del nodo y devuélvalo.
        # AYUDA: lista.pop() quita de la lista su último elemento y lo devuelve.
        ############################################################################
        if nodo.sin_probar == []:
            return nodo
        else:
            accion, sucesor = nodo.sin_probar.pop()
            hijo = Nodo(self.game, sucesor, padre=nodo, accion=accion)
            nodo.hijos.append(hijo)
            return hijo

    # 3 · SIMULACIÓN: una partida al azar, SIN heurística
    def simular(self, state):
        ############################################################################
        # EJERCICIO 3 c): SE DEBE COMPLETAR EL CÓDIGO POR EL ESTUDIANTE.
        # Mientras la partida no haya terminado, elija AL AZAR una de las parejas
        # (acción, sucesor) que da self.game.successors(state) y pase a ese sucesor.
        # Al final, devuelva la utilidad del estado terminal (+1, 0 o -1, para MAX).
        # AYUDA: random.choice(lista) devuelve un elemento de la lista, al azar.
        #        Llámela UNA vez por jugada, con la lista que da successors: con la
        #        semilla que fija el programa obtendrá exactamente los números del
        #        guion.
        ############################################################################
        while not self.game.is_end(state):
            accion, sucesor = random.choice(self.game.successors(state))
            state = sucesor
        return self.game.utility(state)

    # 4 · RETROPROPAGACIÓN: el resultado sube hasta la raíz
    def retropropagar(self, nodo, resultado):
        ############################################################################
        # EJERCICIO 3 d): SE DEBE COMPLETAR EL CÓDIGO POR EL ESTUDIANTE.
        # Suba desde nodo hasta la raíz, de padre en padre (el padre de la raíz es
        # None). En cada nodo del camino:
        #   - sume un playout a N;
        #   - si tiene padre, sume a U la victoria de QUIEN ELIGIÓ el nodo, que es el
        #     jugador al que le tocaba mover en el PADRE (self.game.player). El
        #     resultado viene visto por MAX: si quien eligió es MIN, cámbiele el
        #     signo. Con ese valor v, visto por quien eligió, U suma (v + 1) / 2:
        #     1 si gana, 1/2 si son tablas y 0 si pierde.
        ############################################################################
        while nodo is not None:
            nodo.N += 1
            if nodo.padre is not None:
                jugador = self.game.player(nodo.padre.state)
                v = resultado if jugador == 'MAX' else -resultado
                nodo.U += (v + 1) / 2
            nodo = nodo.padre
        


class Nodo:
    def __init__(self, game, state, padre=None, accion=None):
        self.state  = state
        self.padre  = padre
        self.accion = accion      # la jugada que llevó hasta aquí
        self.hijos  = []
        self.N      = 0           # playouts que pasaron por aquí
        self.U      = 0           # victorias de quien lo eligió
        # jugadas aún sin probar (ninguna, si es terminal)
        self.sin_probar = []
        if not game.is_end(state):
            self.sin_probar = game.successors(state)
