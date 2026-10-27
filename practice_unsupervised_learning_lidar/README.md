# Práctica 4 (LiDAR). Aprendizaje no supervisado: k-means y DBSCAN sobre nubes de puntos

Agrupamiento de barridos reales de un LiDAR Velodyne HDL-64E (KITTI) para separar el suelo
del resto y encontrar objetos. Las etiquetas de SemanticKITTI NO se usan para agrupar: solo
para medir el resultado. El guion está en
`Intro2AI_lessons/lesson4-unsupervised-learning-lidar/main.tex`.

    pip install -r requirements.txt
    python 1_visualizar_main.py     # ejercicio 1: leer y ver un barrido (lidar/nube.py)
    python 2_kmeans_main.py         # ejercicio 2: k-means desde cero (lidar/kmeans.py)
    python 3_suelo_main.py          # ejercicio 3: suelo frente al resto, k = 2, tres juegos de atributos
    python 4_dbscan_main.py         # ejercicio 4: objetos con DBSCAN

Datos (`data/`): barridos 00/001560 y 08/003880 de KITTI odometría (Geiger et al., 2012,
CC BY-NC-SA 3.0) con sus etiquetas de SemanticKITTI (Behley et al., 2019, CC BY-NC-SA 4.0).
