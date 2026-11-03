"""
    HERRAMIENTAS DE LA PRÁCTICA. Se dan completas: no hay que modificar nada.

    - comprobar(...): compara lo obtenido con lo esperado y escribe [BIEN] o [FALLA]
      (con una pista si falla). resumen() cuenta cuántas han salido bien.
    - intentar(...): ejecuta un apartado de un programa; si falta algún ejercicio por
      completar (NotImplementedError) o salta un error, lo dice y sigue con el siguiente.
    - figura_campanas(...): la figura del ejercicio 1, que se guarda en figures/.
"""
import os
import traceback
import numpy as np

AZUL, VERDE, ROJO = '#1d4e89', '#1c7a4d', '#b4451f'
RESULTADOS = {'bien': 0, 'falla': 0}


def miles(n):
    """4180 -> '4.180', para leer los números grandes de un vistazo."""
    return '{:,}'.format(int(n)).replace(',', '.')


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
# La figura del ejercicio 1
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


def figura_campanas(X, y, medias, desviaciones, clases, atributos, ruta):
    """
    Ejercicio 1: una gráfica por atributo, con el histograma de cada clase y, encima, la
    campana de Gauss que ha aprendido el clasificador para ella (con su media y su
    desviación típica).
    """
    import matplotlib.pyplot as plt
    X, y = np.asarray(X, dtype=float), np.asarray(y)
    fig, ejes = plt.subplots(1, len(atributos), figsize=(11, 2.9))
    for i, (ax, nombre) in enumerate(zip(ejes, atributos)):
        rejilla = np.linspace(X[:, i].min() - 0.3, X[:, i].max() + 0.3, 400)
        altura = 0
        for c, (clase, color) in enumerate(zip(clases, [AZUL, VERDE, ROJO])):
            ax.hist(X[y == c, i], bins=8, density=True, color=color, alpha=0.25)
            media, desviacion = medias[c][i], desviaciones[c][i]
            campana = (np.exp(-(rejilla - media) ** 2 / (2 * desviacion ** 2))
                       / (np.sqrt(2 * np.pi) * desviacion))
            ax.plot(rejilla, campana, color=color, lw=2, label=clase)
            altura = max(altura, campana.max())
        ax.set_ylim(0, 1.3 * altura)       # las campanas, enteras; alguna barra puede salirse
        ax.set_title('$x_%d$: %s' % (i + 1, nombre), fontsize=10)
        ax.set_xlabel('cm', fontsize=9)
        ax.tick_params(labelsize=8)
    ejes[0].set_ylabel('densidad', fontsize=9)
    ejes[0].legend(fontsize=8)
    _guardar(fig, ruta)
