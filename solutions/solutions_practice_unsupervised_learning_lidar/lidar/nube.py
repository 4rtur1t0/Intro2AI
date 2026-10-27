"""
    Lectura, recorte, coloreado y visualización de nubes de puntos LiDAR.

    Una nube de KITTI es un fichero .bin con números float32 seguidos, cuatro por punto:
    x, y, z (en metros, en el sistema del sensor) y la intensidad (entre 0 y 1).
        x: hacia delante del coche,  y: hacia la izquierda,  z: hacia arriba.

    Sus etiquetas (SemanticKITTI) están en un fichero .label con un entero uint32 por punto,
    en el mismo orden que la nube: los 16 bits bajos son la CLASE (coche, calzada, ...) y los
    16 altos, la INSTANCIA (qué coche concreto es).
"""
import numpy as np
import open3d as o3d
from matplotlib import colormaps

# gris para los puntos de ruido de DBSCAN (etiqueta -1)
GRIS_RUIDO = np.array([0.6, 0.6, 0.6])

# clases de SemanticKITTI: nombre y color oficial (RGB, 0-255)
CLASES = {
    0: ('sin etiqueta', (0, 0, 0)),         1: ('valor atípico', (255, 0, 0)),
    10: ('coche', (100, 150, 245)),         11: ('bicicleta', (100, 230, 245)),
    13: ('autobús', (100, 80, 250)),        15: ('moto', (30, 60, 150)),
    16: ('tranvía/tren', (0, 0, 255)),      18: ('camión', (80, 30, 180)),
    20: ('otro vehículo', (0, 0, 255)),     30: ('persona', (255, 30, 30)),
    31: ('ciclista', (255, 40, 200)),       32: ('motorista', (150, 30, 90)),
    40: ('calzada', (255, 0, 255)),         44: ('aparcamiento', (255, 150, 255)),
    48: ('acera', (75, 0, 75)),             49: ('otro suelo', (175, 0, 75)),
    50: ('edificio', (255, 200, 0)),        51: ('valla', (255, 120, 50)),
    52: ('otra estructura', (255, 150, 0)), 60: ('marca vial', (150, 255, 170)),
    70: ('vegetación', (0, 175, 0)),        71: ('tronco', (135, 60, 0)),
    72: ('terreno', (150, 240, 80)),        80: ('poste', (255, 240, 150)),
    81: ('señal de tráfico', (255, 0, 0)),  99: ('otro objeto', (50, 255, 255)),
}
# las clases "en movimiento" (252: coche en movimiento, ...) se tratan como su clase quieta
EN_MOVIMIENTO = {252: 10, 253: 31, 254: 30, 255: 32, 256: 16, 257: 13, 258: 18, 259: 20}


def cargar_nube(ruta):
    """
    Lee un barrido LiDAR guardado en formato .bin de KITTI.
    Devuelve un array de numpy de forma (N, 4): columnas x, y, z, intensidad.
    """
    nube = np.fromfile(ruta, dtype=np.float32)
    nube = np.reshape(nube, [-1, 4])
    return nube


def cargar_etiquetas(ruta):
    """
    Lee las etiquetas de SemanticKITTI (.label) de un barrido.
    Devuelve dos vectores (N,): la clase y la instancia de cada punto.
    """
    etiquetas = np.fromfile(ruta, dtype=np.uint32)
    clase = (etiquetas & 0xFFFF).astype(np.int64)
    instancia = (etiquetas >> 16).astype(np.int64)
    for movil, quieta in EN_MOVIMIENTO.items():
        clase[clase == movil] = quieta
    return clase, instancia


def dentro_del_recorte(nube, lado=20.0):
    """
    Vector booleano (N,): True para los puntos que caen en un cuadrado de 2*lado x 2*lado
    metros centrado en el sensor (|x| < lado y |y| < lado). Sirve para recortar a la vez la
    nube y sus etiquetas:  m = dentro_del_recorte(nube);  nube, clase = nube[m], clase[m]
    """
    return (np.abs(nube[:, 0]) < lado) & (np.abs(nube[:, 1]) < lado)


def colores_por_valor(valores, mapa='viridis'):
    """
    Convierte un valor por punto (altura, intensidad, ...) en un color RGB por punto.
    Se recortan los extremos (percentiles 2 y 98) para que unos pocos puntos raros
    no se lleven toda la escala de color. Devuelve un array (N, 3) con valores en [0, 1].
    """
    lo, hi = np.percentile(valores, [2, 98])
    normalizados = np.clip((valores - lo) / (hi - lo + 1e-12), 0, 1)
    return colormaps[mapa](normalizados)[:, :3]


def colores_por_clase(clase):
    """Color oficial de SemanticKITTI para la clase de cada punto. Devuelve (N, 3) en [0, 1]."""
    colores = np.zeros((len(clase), 3))
    for c, (_, rgb) in CLASES.items():
        colores[clase == c] = np.array(rgb) / 255.0
    return colores


def colores_por_grupo(etiquetas, semilla=0):
    """
    Un color distinto para cada grupo. La etiqueta -1 (ruido de DBSCAN) se pinta en gris.
    Los colores se eligen al azar para que dos grupos vecinos casi nunca se parezcan.
    """
    etiquetas = np.asarray(etiquetas)
    rng = np.random.default_rng(semilla)
    n_grupos = max(etiquetas.max() + 1, 1)
    paleta = rng.uniform(0.1, 0.9, size=(n_grupos, 3))
    colores = np.tile(GRIS_RUIDO, (len(etiquetas), 1))
    validos = etiquetas >= 0
    colores[validos] = paleta[etiquetas[validos]]
    return colores


def a_open3d(puntos, colores=None):
    """Crea un o3d.geometry.PointCloud con las tres primeras columnas de `puntos`."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(puntos[:, :3].astype(np.float32))
    if colores is not None:
        pcd.colors = o3d.utility.Vector3dVector(colores)
    return pcd


def visualizar(puntos, colores=None, titulo='Nube LiDAR'):
    """
    Abre una ventana de open3d con la nube. El programa se detiene hasta que se cierra.
    Se dibujan también unos ejes de 2 m en el sensor (rojo = x, verde = y, azul = z).
    Ratón: botón izquierdo gira, rueda acerca, Ctrl + izquierdo desplaza.
    Teclado: '+' / '-' cambian el tamaño de los puntos.
    """
    ejes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
    o3d.visualization.draw_geometries([a_open3d(puntos, colores), ejes],
                                      window_name=titulo, width=1280, height=800)
