"""
Navigate to 
overpass-turbo.eu
execute this query.
Use export --> raw data OSM

MAP 0: Small map. Test map
[out:xml];
(
  way["highway"]({{bbox}});
);
out body;
>;
out skel qt;

MAP 2: larger area, less detail.
[out:xml][timeout:90];
(
  way["highway"~"^(motorway|trunk|primary|secondary)(_link)?$"]({{bbox}});
);
out body;
>;
out skel qt;
"""
from search_algorithms.a_star_algorithm_osm import A_star_Algorithm_OSM
# import matplotlib.pyplot as plt
import osmnx as ox


if __name__ == "__main__":
    # OPCIÓN A: Descargar datos en vivo (Recomendado)
    # lugar = "Elche, Spain"
    # print(f"Descargando la red vial para: {lugar}...")
    # G = ox.graph_from_place(lugar, network_type="drive")
    # OPCIÓN B: Usar un archivo local descargado de Overpass Turbo
    ################################
    print('Cargando el mapa...')
    G = ox.graph_from_xml("maps/map0.osm")
    print('Mapa cargado.')
    gps_origen = (38.2694, -0.706661)
    gps_destino = (38.25, -0.65)
    ################################
    ################################
    # print('Cargando el mapa...')
    # G = ox.graph_from_xml("maps/map1.osm")
    # print('Mapa cargado.')
    # gps_origen = (38.2694, -0.706661)
    # gps_destino = (38.35, -0.48)
    ################################
    ###############################
    # print('Cargando el mapa...')
    # G = ox.graph_from_xml("maps/map2.osm")
    # print('Mapa cargado.')
    # gps_origen = (38.2694, -0.706661)
    # gps_destino = (39.35, -0.48)
    ################################


    # EL MISMO ALGORITMO DE ANTES --> se debe modificar para manejar el nuevo mapa
    algoritmo = A_star_Algorithm_OSM(G)
    # Buscamos los nodos reales de la red más cercanos a nuestros puntos GPS
    nodo_origen = algoritmo.get_closest_node(gps_origen)
    nodo_destino = algoritmo.get_closest_node(gps_destino)
    print(f"ID Nodo Origen más cercano: {nodo_origen}")
    print(f"ID Nodo Destino más cercano: {nodo_destino}")
    route, distance, iterations = algoritmo.find_route(nodo_origen, nodo_destino)
    # Ahora ploteamos sobre un mapa real satelital. Ajusta el zoom entre 5 (grueso) y 16 (fino)
    algoritmo.plot_route(route, distance, zoom_level=13)
