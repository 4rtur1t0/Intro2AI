"""
    Características geométricas de cada punto (YA PROGRAMADO: no hay que tocar nada).

    Para cada punto se toman sus vecinos a menos de `radio` metros y se calcula la matriz de
    covarianza 3x3 de sus coordenadas. Sus autovectores son las direcciones principales del
    vecindario (un PCA local) y sus autovalores l1 >= l2 >= l3 dicen cuánto se extienden los
    puntos en cada una de ellas:
        - la NORMAL de la superficie es la dirección en la que el vecindario varía MENOS,
          el autovector de l3;
        - VERTICALIDAD = 1 - |n_z|: 0 en superficies horizontales (calzada, techo de un coche)
          y 1 en verticales (paredes, laterales de los coches, troncos);
        - CURVATURA = l3 / (l1 + l2 + l3): 0 en superficies lisas y hasta 1/3 en vecindarios
          desordenados (vegetación, esquinas).
"""
import numpy as np
import open3d as o3d


def caracteristicas_geometricas(nube, radio=0.5, max_vecinos=100):
    """
    Devuelve dos vectores (N,): la verticalidad y la curvatura de cada punto de la nube.
    Para 100.000 puntos tarda menos de un segundo.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(nube[:, :3].astype(np.float64))
    pcd.estimate_covariances(o3d.geometry.KDTreeSearchParamHybrid(radius=radio, max_nn=max_vecinos))
    # eigh devuelve los autovalores en orden CRECIENTE: la columna 0 es l3
    autovalores, autovectores = np.linalg.eigh(np.asarray(pcd.covariances))
    normales = autovectores[:, :, 0]
    verticalidad = 1.0 - np.abs(normales[:, 2])
    curvatura = autovalores[:, 0] / np.maximum(autovalores.sum(axis=1), 1e-12)
    return verticalidad, curvatura
