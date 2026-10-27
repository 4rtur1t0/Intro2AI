"""
    PRÁCTICA 4 (LiDAR). APRENDIZAJE NO SUPERVISADO. EJERCICIO 3: SUELO FRENTE AL RESTO.

    ¿Es capaz k-means con k = 2 de separar el suelo de todo lo demás? El algoritmo es siempre
    el mismo; lo único que cambia es QUÉ ATRIBUTOS le damos de cada punto:
        a) sus coordenadas (x, y, z),
        b) solo su altura z,
        c) su altura más dos características geométricas: verticalidad y curvatura.
    El resultado se mide con las etiquetas de SemanticKITTI (lidar/suelo.py): matriz de
    confusión, accuracy, precision, recall, F1-score e IoU, con el suelo como clase positiva.

    Completar: los huecos de este fichero.
"""
import numpy as np
from lidar.nube import (cargar_nube, cargar_etiquetas, dentro_del_recorte, colores_por_valor,
                        colores_por_grupo, visualizar)
from lidar.kmeans import kmeans
from lidar.suelo import (es_suelo, prediccion_de_suelo, matriz_de_confusion, metricas,
                         imprimir_evaluacion, CLASES_DUDOSAS)
from lidar.caracteristicas import caracteristicas_geometricas

BARRIDO = 'data/kitti_00_001560'


def probar(nombre, X, nube, clase):
    """Agrupa X con k-means (k = 2), lo evalúa con las etiquetas y lo muestra en open3d."""
    etiquetas, C, _, _ = kmeans(X, 2, semilla=0)
    prediccion = prediccion_de_suelo(etiquetas, clase)
    matriz = matriz_de_confusion(prediccion, clase)
    imprimir_evaluacion(*matriz)
    # mismos colores en todas las ventanas: suelo predicho = grupo 0
    visualizar(nube, colores_por_grupo((~prediccion).astype(int)),
               titulo='Ejercicio 3 · k-means con %s, k = 2' % nombre)
    return C, metricas(*matriz)


if __name__ == "__main__":
    nube = cargar_nube(BARRIDO + '.bin')
    clase, _ = cargar_etiquetas(BARRIDO + '.label')
    m = dentro_del_recorte(nube)
    nube, clase = nube[m], clase[m]
    validos = ~np.isin(clase, CLASES_DUDOSAS)
    print('Puntos en el recorte: %d (%d con etiqueta válida)   ·   suelo según las etiquetas: %.1f %%'
          % (len(nube), validos.sum(), 100 * es_suelo(clase)[validos].mean()))
    visualizar(nube, colores_por_grupo((~es_suelo(clase)).astype(int)),
               titulo='Ejercicio 3 · el suelo según las etiquetas')
    tabla = {}

    # referencia trivial: decir que TODO es suelo, sin mirar nada
    print('\nReferencia trivial: todo es suelo')
    matriz = matriz_de_confusion(np.ones(len(nube), dtype=bool), clase)
    imprimir_evaluacion(*matriz)
    tabla['todo es suelo'] = metricas(*matriz)

    # ------------------------------------------------------------------
    # 3.a Atributos: las tres coordenadas (x, y, z)
    # ------------------------------------------------------------------
    X_xyz = nube[:, :3].astype(np.float64)
    print('\n3.a  atributos (x, y, z)')
    _, tabla['(x, y, z)'] = probar('(x, y, z)', X_xyz, nube, clase)

    # ------------------------------------------------------------------
    # 3.b Atributo: solo la altura z
    # ------------------------------------------------------------------
    X_z = nube[:, 2:3].astype(np.float64)
    print('\n3.b  atributo z')
    C, tabla['z'] = probar('z', X_z, nube, clase)
    print('     centroides (z): %.2f y %.2f m' % tuple(C.ravel()))
    umbral = C.mean()
    print('     frontera entre los dos grupos: z = %.2f m' % umbral)

    # ------------------------------------------------------------------
    # 3.c Atributos: z + características geométricas (ya calculadas)
    # ------------------------------------------------------------------
    print('\n3.c  atributos z + verticalidad + curvatura (estandarizados)')
    verticalidad, curvatura = caracteristicas_geometricas(nube)
    visualizar(nube, colores_por_valor(verticalidad, 'coolwarm'),
               titulo='Ejercicio 3.c · verticalidad (azul = horizontal, rojo = vertical)')
    visualizar(nube, colores_por_valor(curvatura, 'magma'),
               titulo='Ejercicio 3.c · curvatura (oscuro = liso, claro = desordenado)')
    ######################################################
    # CARACTERÍSTICAS: (z, vert, curv)
    ######################################################
    z = nube[:, 2:3].astype(np.float64)
    X_geo = np.column_stack((z, verticalidad, curvatura)).astype(np.float64)
    # estandarizar: cada columna con media 0 y desviación 1, para que las tres pesen igual
    X_geo = (X_geo - X_geo.mean(axis=0)) / X_geo.std(axis=0)
    _, tabla['z + geometría'] = probar('z + verticalidad + curvatura', X_geo, nube, clase)

    ######################################################
    # OPCIONAL: que las pruebe el alumno otras características: (vert, curv)
    # OPCIONAL: que las pruebe el alumno otras características: (z, vert, curv, Intensidad)
    # OPCIONAL: que las pruebe el alumno: (curv)
    # OPCIONAL: que las pruebe el alumno: (vert)
    # OPCIONAL: que las pruebe el alumno: (z, vert)
    # OPCIONAL: que las pruebe el alumno: (z, curv)
    # OPCIONAl: (2*vert, curv). Se pueden escalar después de la normalizacíón
    ######################################################
    I = nube[:, 3:4].astype(np.float64)
    X_geo = np.column_stack((z, verticalidad, curvatura, I)).astype(np.float64)
    # X_geo = np.column_stack((verticalidad, curvatura)).astype(np.float64)
    # X_geo = np.column_stack([curvatura]).astype(np.float64)
    # X_geo = np.column_stack([verticalidad]).astype(np.float64)
    # X_geo = np.column_stack((z, verticalidad)).astype(np.float64)
    # X_geo = np.column_stack((z, curvatura)).astype(np.float64)
    # estandarizar: cada columna con media 0 y desviación 1, para que las tres pesen igual
    X_geo = (X_geo - X_geo.mean(axis=0)) / X_geo.std(axis=0)
    _, tabla['otra geometría'] = probar('otras caracteristicas', X_geo, nube, clase)

    # RESUMEN DE RESULTADOS
    print('\nRESUMEN (suelo = clase positiva)  accuracy  precision  recall  F1-score    IoU')
    for nombre, m in tabla.items():
        print('   %-30s' % nombre + ''.join('%9.1f %%' % (100 * v) for v in m.values()))
