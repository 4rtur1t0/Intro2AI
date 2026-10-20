# Práctica 2.2. Búsqueda con adversario: minimax, poda alfa-beta y MCTS

**Versión resuelta** (para el profesor). La del alumno está en `Intro2AI/practice_adversarial_search_reviewed/`.

Se programan **desde cero**, en python puro, tres algoritmos de búsqueda con adversario: minimax,
la poda alfa-beta y Monte Carlo Tree Search (MCTS), y se ponen a prueba en el **tres en raya**.
Opcionalmente, el mismo MCTS juega a **Conecta 4**, donde ya no se puede llegar al final de la partida.

El guion está en `Intro2AI_lessons/lesson2-adversarial-search_reviewed/main.tex`.

    pip install -r requirements.txt
    python 1_minimax_main.py           # ejercicio 1: minimax
    python jugar.py --rival minimax    #   y jugar contra su minimax
    python 2_alfabeta_main.py          # ejercicio 2: poda alfa-beta
    python 3_mcts_main.py              # ejercicio 3: MCTS, paso a paso
    python jugar.py --rival mcts       #   y jugar contra su MCTS (python jugar.py --help)
    python jugar.py --juego conecta4   # opcional: Conecta 4 contra su MCTS

| Fichero | ¿Se completa? | Contenido |
|---|---|---|
| `juegos/tres_en_raya.py` | no | `TresEnRaya`: las reglas del tres en raya, escritas como una clase |
| `juegos/conecta4.py` | no | `Conecta4`, con la misma interfaz (opcional) |
| `algoritmos/minimax.py` | ejercicio 1 | `Minimax`: `valor`, `valor_max` y `valor_min` |
| `algoritmos/alfabeta.py` | ejercicio 2 | `AlfaBeta`: los mismos métodos, con las cotas alfa y beta |
| `algoritmos/mcts.py` | ejercicio 3 | `MCTS`: `ucb1` y `seleccionar` (a), `expandir` (b), `simular` (c) y `retropropagar` (d) |
| `herramientas.py` | no | comprobaciones, el árbol de MCTS (`construir_arbol` y `dibujar_arbol`), el jugador humano y la figura del ejercicio 2 |

Los huecos están marcados con `EJERCICIO N: SE DEBE COMPLETAR EL CÓDIGO POR EL ESTUDIANTE` (en el
ejercicio 3, `EJERCICIO 3 a)`, `b)`...). Mientras no se completan, lanzan `NotImplementedError`. Cada
programa comprueba sus resultados y escribe [BIEN] o [FALLA]. Los programas son cortos y cada
apartado del guion es una función: léalos para saber qué hace y qué comprueba cada uno. La figura
del ejercicio 2 se guarda en `figures/`. Hay que ejecutar los programas **desde este directorio**.
