"""
    PRÁCTICA 2.2. JUEGUE CONTRA LA MÁQUINA.

        python jugar.py --rival minimax                  contra su Minimax (ejercicio 1)
        python jugar.py                                  contra su AlfaBeta (ejercicio 2)
        python jugar.py --rival mcts                     contra su MCTS (ejercicio 3)
        python jugar.py --rival mcts --iteraciones 100   MCTS con 100 iteraciones por jugada
        python jugar.py --empieza humano                 empieza usted, con X
        python jugar.py --juego conecta4                 opcional: Conecta 4 contra su MCTS
        python jugar.py --juego conecta4 --iteraciones 2000

    Contra minimax o alfa-beta no se puede ganar: si alguna vez lo consigue, hay un
    error en su código.

    En Conecta 4 solo juega MCTS: minimax y alfa-beta tendrían que llegar al final de
    la partida, y el árbol de Conecta 4 tiene del orden de 10^21 nodos.
"""
import argparse
from juegos import tres_en_raya, conecta4
from algoritmos.minimax import Minimax
from algoritmos.alfabeta import AlfaBeta
from algoritmos.mcts import MCTS
from herramientas import Humano, nombre_de


def la_maquina(args, game):
    if args.rival == 'mcts':
        return MCTS(game, args.iteraciones)
    if args.rival == 'minimax':
        return Minimax(game)
    return AlfaBeta(game)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--juego', default='tresenraya', choices=['tresenraya', 'conecta4'])
    parser.add_argument('--rival', choices=['alfabeta', 'minimax', 'mcts'],
                        help='por defecto, alfabeta en el tres en raya y mcts en Conecta 4')
    parser.add_argument('--iteraciones', type=int, default=1000, help='iteraciones de MCTS')
    parser.add_argument('--empieza', default='maquina', choices=['maquina', 'humano'])
    args = parser.parse_args()

    if args.juego == 'conecta4':
        if args.rival in ('alfabeta', 'minimax'):
            print('En Conecta 4 solo juega MCTS: %s tendría que llegar al final de la '
                  'partida, y el árbol tiene del orden de 10^21 nodos.'
                  % {'alfabeta': 'alfa-beta', 'minimax': 'minimax'}[args.rival])
            raise SystemExit(1)
        args.rival = 'mcts'
        modulo, game = conecta4, conecta4.Conecta4()
    else:
        args.rival = args.rival or 'alfabeta'
        modulo, game = tres_en_raya, tres_en_raya.TresEnRaya()
    maquina = la_maquina(args, game)
    try:
        maquina.decidir(game.start_state())      # ¿sabe ya jugar la máquina?
    except NotImplementedError as error:
        print('La máquina todavía no sabe jugar: %s' % error)
        raise SystemExit(1)
    humano = Humano(game, modulo.dibujar)
    if args.empieza == 'maquina':
        jugador_x, jugador_o = maquina, humano
        print('La máquina (%s) juega con X y empieza; usted juega con O.' % nombre_de(maquina))
    else:
        jugador_x, jugador_o = humano, maquina
        print('Usted juega con X y empieza; la máquina (%s) juega con O.' % nombre_de(maquina))

    state = game.start_state()
    while not game.is_end(state):
        turno = 'X' if game.player(state) == "MAX" else 'O'
        jugador = jugador_x if turno == 'X' else jugador_o
        accion = jugador.decidir(state)
        if jugador is maquina:
            print('La máquina (%s) juega en la %s %d.'
                  % (turno, 'columna' if args.juego == 'conecta4' else 'casilla', accion))
        state = dict(game.successors(state))[accion]
    modulo.dibujar(state)
    u = game.utility(state)
    print('Tablas.' if u == 0 else 'Gana %s.' % ('X' if u > 0 else 'O'))
