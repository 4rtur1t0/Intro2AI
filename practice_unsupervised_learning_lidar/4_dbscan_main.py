"""
    PRÁCTICA 4 (LiDAR). APRENDIZAJE NO SUPERVISADO. EJERCICIO 4: OBJETOS CON DBSCAN.

    Con el suelo ya quitado, se agrupan los puntos que quedan en vista cenital (x, y) con
    DBSCAN, que no necesita saber cuántos objetos hay. Después se compara con k-means.
    Las etiquetas de instancia de SemanticKITTI dicen qué puntos son de cada coche: con ellas
    se mide, con el criterio estándar de IoU > 0,5, cuántos coches detecta cada algoritmo.

    Completar: la función agrupar_dbscan de este fichero.
"""
import numpy as np
from sklearn.cluster import DBSCAN
from lidar.nube import cargar_nube, cargar_etiquetas, dentro_del_recorte, colores_por_grupo, visualizar
from lidar.kmeans import kmeans
from lidar.suelo import es_suelo, es_objeto

BARRIDO = 'data/kitti_00_001560'
EPS, MIN_SAMPLES = 0.5, 8
CLASE_COCHE = 10


def agrupar_dbscan(X, eps, min_samples):
    """
    Agrupa X con DBSCAN de scikit-learn.
    Devuelve (etiquetas, número de grupos, número de puntos de ruido).
    Recuerde: DBSCAN marca el ruido con la etiqueta -1, que NO es un grupo.
    """
    ############################################################################
    # EJERCICIO 4: SE DEBE COMPLETAR EL CÓDIGO POR EL ESTUDIANTE.
    # AYUDA: DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    ############################################################################
    raise NotImplementedError('EJERCICIO 4 sin completar: agrupar_dbscan()')


def coches_detectados(grupos, clase, instancia, iou_minimo=0.5, min_puntos=50):
    """
    Evalúa un agrupamiento con las instancias de coche de las etiquetas, con el criterio
    estándar de la segmentación panóptica (Kirillov et al., 2019), el que usa SemanticKITTI:
    un coche está DETECTADO si algún grupo tiene con él un IoU > 0,5, donde
        IoU = |puntos del coche ∩ puntos del grupo| / |puntos del coche ∪ puntos del grupo|.
    Con IoU > 0,5 cada coche solo puede emparejarse con un grupo. Como en SemanticKITTI, se
    ignoran los coches con menos de 50 puntos (los muy lejanos o tapados).
    Devuelve (detectados, coches, recall = detectados / coches, IoU medio de los detectados).
    """
    coche = clase == CLASE_COCHE
    ids = [i for i in np.unique(instancia[coche]) if np.sum(coche & (instancia == i)) >= min_puntos]
    ious = []
    for i in ids:
        suyo = coche & (instancia == i)
        mejor = 0.0
        for g, n in zip(*np.unique(grupos[suyo], return_counts=True)):
            if g >= 0:                                   # el ruido (-1) no es un grupo
                mejor = max(mejor, n / (suyo.sum() + np.sum(grupos == g) - n))
        if mejor > iou_minimo:
            ious.append(mejor)
    recall = len(ious) / len(ids) if ids else 0.0
    return len(ious), len(ids), recall, (np.mean(ious) if ious else 0.0)


def resumen_coches(grupos, clase, instancia):
    """Texto con la evaluación de coches_detectados."""
    detectados, total, recall, iou = coches_detectados(grupos, clase, instancia)
    return ('coches detectados (IoU > 0,5): %d de %d  ->  recall %.1f %%, IoU medio %.2f'
            % (detectados, total, 100 * recall, iou))


def ver_con_suelo(objetos, etiquetas, suelo, titulo):
    """Muestra los objetos coloreados por grupo y, debajo, el suelo en gris claro."""
    puntos = np.vstack([objetos[:, :3], suelo[:, :3]])
    colores = np.vstack([colores_por_grupo(etiquetas), np.full((len(suelo), 3), 0.85)])
    visualizar(puntos, colores, titulo=titulo)


if __name__ == "__main__":
    nube = cargar_nube(BARRIDO + '.bin')
    clase, instancia = cargar_etiquetas(BARRIDO + '.label')
    m = dentro_del_recorte(nube)
    nube, clase, instancia = nube[m], clase[m], instancia[m]
    obj = es_objeto(nube, clase)
    objetos, clase_obj, instancia_obj = nube[obj], clase[obj], instancia[obj]
    suelo = nube[es_suelo(clase)]
    XY = objetos[:, :2].astype(np.float64)
    print('Puntos en el recorte: %d   ·   objetos (sin suelo, z < 0,8 m): %d' % (len(nube), len(objetos)))

    # 4.a DBSCAN en vista cenital
    etiquetas, n_grupos, n_ruido = agrupar_dbscan(XY, EPS, MIN_SAMPLES)
    print('\n4.a  DBSCAN (eps = %.1f m, min_samples = %d): %d grupos, %d puntos de ruido'
          % (EPS, MIN_SAMPLES, n_grupos, n_ruido))
    print('     ' + resumen_coches(etiquetas, clase_obj, instancia_obj))
    ver_con_suelo(objetos, etiquetas, suelo, 'Ejercicio 4.a · DBSCAN: %d grupos (ruido en gris)' % n_grupos)

    # 4.b ¿Cómo influye eps?
    print('\n4.b  Efecto de eps (min_samples = %d)' % MIN_SAMPLES)
    for eps in [0.2, 0.3, 0.5, 1.0, 2.0]:
        e, g, r = agrupar_dbscan(XY, eps, MIN_SAMPLES)
        print('     eps = %.1f m: %3d grupos, %4d de ruido, %s'
              % (eps, g, r, resumen_coches(e, clase_obj, instancia_obj)))

    # 4.c k-means con el mismo número de grupos que ha encontrado DBSCAN
    etiquetas_km, _, _, _ = kmeans(XY, n_grupos, semilla=0)
    print('\n4.c  k-means con k = %d' % n_grupos)
    print('     ' + resumen_coches(etiquetas_km, clase_obj, instancia_obj))
    ver_con_suelo(objetos, etiquetas_km, suelo, 'Ejercicio 4.c · k-means, k = %d' % n_grupos)

    # 4.d DBSCAN SIN quitar el suelo, en 3D
    etiquetas_3d, g, r = agrupar_dbscan(nube[:, :3].astype(np.float64), EPS, MIN_SAMPLES)
    mayor = np.bincount(etiquetas_3d[etiquetas_3d >= 0]).max()
    print('\n4.d  DBSCAN con suelo, en (x, y, z): %d grupos, %d de ruido; '
          'el grupo mayor tiene el %.1f %% de los puntos' % (g, r, 100 * mayor / len(nube)))
    visualizar(nube, colores_por_grupo(etiquetas_3d), titulo='Ejercicio 4.d · DBSCAN sin quitar el suelo')
