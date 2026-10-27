"""
    PRÁCTICA 4 (LiDAR). APRENDIZAJE NO SUPERVISADO. EJERCICIO 1: LEER Y VER UN BARRIDO.

    Se lee un barrido real de un LiDAR Velodyne HDL-64E (KITTI) y se visualiza en 3D con
    open3d, coloreando cada punto según la intensidad con que volvió el láser y según su
    clase (etiquetas de SemanticKITTI, puestas a mano por personas).

    Completar: lidar/nube.py (cargar_nube, a_open3d) y los huecos de este fichero.
"""
import numpy as np
from lidar.nube import (cargar_nube, cargar_etiquetas, colores_por_valor, colores_por_clase,
                        visualizar, CLASES)

BARRIDO = 'data/kitti_00_001560'


if __name__ == "__main__":
    ############################################################################
    # EJERCICIO 1: SE DEBE COMPLETAR EL CÓDIGO POR EL ESTUDIANTE.
    # Complete la función cargar_nube para cargar los puntos de la nube
    ############################################################################
    nube = cargar_nube(BARRIDO + '.bin')
    print('Forma del array leído: ', nube.shape)
    print('Primeros 5 puntos (x, y, z, intensidad):')
    print(nube[:5])

    ############################################################################
    # EJERCICIO 1: SE DEBE COMPLETAR EL CÓDIGO POR EL ESTUDIANTE.
    # Calcule el número de puntos del barrido y extraiga la columna de intensidad.
    ############################################################################
    raise NotImplementedError('EJERCICIO 1 sin completar: n_puntos e intensidad')
    # print('Número de puntos del barrido: %d' % n_puntos)
    # print('Intensidad: mínima %.3f, media %.3f, máxima %.3f'
    #       % (intensidad.min(), intensidad.mean(), intensidad.max()))
    # visualizar(nube, colores_por_valor(intensidad), titulo='Ejercicio 1 · color = intensidad')

    # las etiquetas: lo que NO usaremos para agrupar, solo para medir
    clase, _ = cargar_etiquetas(BARRIDO + '.label')
    print('\nClases presentes en el barrido (etiquetas de SemanticKITTI):')
    for c, n in sorted(zip(*np.unique(clase, return_counts=True)), key=lambda t: -t[1]):
        print('   %-18s %6d puntos (%4.1f %%)' % (CLASES[c][0], n, 100 * n / len(clase)))
    visualizar(nube, colores_por_clase(clase), titulo='Ejercicio 1 · color = clase (SemanticKITTI)')
