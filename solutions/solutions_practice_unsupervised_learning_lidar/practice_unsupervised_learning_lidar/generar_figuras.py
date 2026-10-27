"""
    Genera las capturas de open3d que usa el guion de la práctica (figures/).
    No forma parte de la práctica del estudiante.

        python generar_figuras.py ../../../Intro2AI_lessons/lesson4-unsupervised-learning-lidar/figures
"""
import os
import sys
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
from lidar.nube import (cargar_nube, cargar_etiquetas, dentro_del_recorte, colores_por_valor,
                        colores_por_clase, colores_por_grupo, a_open3d)
from lidar.kmeans import kmeans
from lidar.suelo import es_suelo, es_objeto, prediccion_de_suelo
from lidar.caracteristicas import caracteristicas_geometricas

BARRIDO = 'data/kitti_00_001560'


def capturar(puntos, colores, fichero, zoom=0.5, front=(-0.55, 0.0, 1.0), tam=2.0):
    """Renderiza la nube sin abrir ventana visible y guarda la imagen."""
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1400, height=1000)
    vis.add_geometry(a_open3d(puntos, colores))
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0))
    vis.get_render_option().point_size = tam
    vc = vis.get_view_control()
    vc.set_lookat([0.0, 0.0, -1.0])
    vc.set_front(list(front))
    vc.set_up([1.0, 0.0, 0.0])
    vc.set_zoom(zoom)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(fichero, do_render=True)
    vis.destroy_window()
    print('figura:', fichero)


if __name__ == "__main__":
    salida = sys.argv[1] if len(sys.argv) > 1 else 'figures'
    os.makedirs(salida, exist_ok=True)
    f = lambda nombre: os.path.join(salida, nombre)

    bruta = cargar_nube(BARRIDO + '.bin')
    clase_bruta, instancia_bruta = cargar_etiquetas(BARRIDO + '.label')
    capturar(bruta, colores_por_valor(bruta[:, 3]), f('ej1_intensidad.png'), zoom=0.18)
    capturar(bruta, colores_por_clase(clase_bruta), f('ej1_clases.png'), zoom=0.18)

    m = dentro_del_recorte(bruta)
    nube, clase, instancia = bruta[m], clase_bruta[m], instancia_bruta[m]
    X = nube[:, :3].astype(np.float64)
    suelo = es_suelo(clase)
    capturar(nube, colores_por_grupo((~suelo).astype(int)), f('ej3_referencia.png'))
    et, _, _, _ = kmeans(X, 2, semilla=0)
    capturar(nube, colores_por_grupo((~prediccion_de_suelo(et, clase)).astype(int)), f('ej3_xyz.png'))
    et, _, _, _ = kmeans(X[:, 2:3], 2, semilla=0)
    capturar(nube, colores_por_grupo((~prediccion_de_suelo(et, clase)).astype(int)), f('ej3_z.png'))
    vert, curv = caracteristicas_geometricas(nube)
    capturar(nube, colores_por_valor(vert, 'coolwarm'), f('ej3_verticalidad.png'))
    capturar(nube, colores_por_valor(curv, 'magma'), f('ej3_curvatura.png'))
    G = np.column_stack([nube[:, 2], vert, curv]).astype(np.float64)
    G = (G - G.mean(axis=0)) / G.std(axis=0)
    et, _, _, _ = kmeans(G, 2, semilla=0)
    capturar(nube, colores_por_grupo((~prediccion_de_suelo(et, clase)).astype(int)), f('ej3_geo.png'))

    obj = es_objeto(nube, clase)
    XY = nube[obj, :2].astype(np.float64)
    puntos = np.vstack([nube[obj, :3], nube[suelo, :3]])
    gris = np.full((suelo.sum(), 3), 0.85)
    et = DBSCAN(eps=0.5, min_samples=8).fit_predict(XY)
    capturar(puntos, np.vstack([colores_por_grupo(et), gris]), f('ej4_dbscan.png'))
    n_grupos = len(set(et) - {-1})
    et, _, _, _ = kmeans(XY, n_grupos, semilla=0)
    capturar(puntos, np.vstack([colores_por_grupo(et), gris]), f('ej4_kmeans.png'))
    et = DBSCAN(eps=0.5, min_samples=8).fit_predict(X)
    capturar(nube, colores_por_grupo(et), f('ej4_dbscan_con_suelo.png'))
