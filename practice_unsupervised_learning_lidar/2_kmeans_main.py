"""
    PRÁCTICA 4 (LiDAR). APRENDIZAJE NO SUPERVISADO. EJERCICIO 2: K-MEANS DESDE CERO.

    Se programa k-means con numpy (lidar/kmeans.py) y se comprueba que da EXACTAMENTE lo
    mismo que el de scikit-learn cuando ambos parten de los mismos centroides.

    Completar: lidar/kmeans.py (asignar, actualizar, inercia, kmeans).
"""
import time
import numpy as np
from sklearn.cluster import KMeans
from lidar.nube import cargar_nube, dentro_del_recorte, colores_por_grupo, visualizar
from lidar.kmeans import inicializar, kmeans

BARRIDO = 'data/kitti_00_001560'
K = 4


if __name__ == "__main__":
    # recorte de 40 m x 40 m alrededor del sensor; atributos: x, y, z
    nube = cargar_nube(BARRIDO + '.bin')
    nube = nube[dentro_del_recorte(nube)]
    X = nube[:, :3].astype(np.float64)
    print('Puntos en el recorte: %d' % len(X))

    C0 = inicializar(X, K, semilla=0)

    inicio = time.perf_counter()
    etiquetas, C, J, iteraciones = kmeans(X, K, C_inicial=C0)
    t_nuestro = time.perf_counter() - inicio

    inicio = time.perf_counter()
    sk = KMeans(n_clusters=K, init=C0, n_init=1, max_iter=100, tol=0,
                algorithm='lloyd').fit(X)
    t_sklearn = time.perf_counter() - inicio

    print('\n                  nuestro k-means    scikit-learn')
    print('inercia          %15.1f %15.1f' % (J, sk.inertia_))
    print('iteraciones      %15d %15d' % (iteraciones, sk.n_iter_))
    print('tiempo (s)       %15.2f %15.2f' % (t_nuestro, t_sklearn))
    print('Puntos con la misma etiqueta en ambos: %.2f %%'
          % (100 * np.mean(etiquetas == sk.labels_)))
    print('Centroides (x, y, z) de nuestro k-means:')
    print(np.round(C, 2))

    visualizar(nube, colores_por_grupo(etiquetas), titulo='Ejercicio 2 · k-means, k = %d' % K)
