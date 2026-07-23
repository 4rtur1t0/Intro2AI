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
import matplotlib.pyplot as plt
import contextily as cx
import osmnx as ox

def plot_route(G, ruta_nodos, distancia_total, zoom_level=15):
    """
    Grafica la ruta sobre mapa satelital usando Contextily.
    :param zoom_level: Int entre 10 y 18. Mayor número = mayor resolución.
    """
    if not ruta_nodos:
        print("No fue posible encontrar una ruta transitable.")
        return

    print("\n" + "=" * 30)
    print(f"Distancia total: {distancia_total:.2f} metros")
    print("=" * 30)
    print(f"Generando mapa satelital (Nivel de Zoom: {zoom_level})...")

    # 1. Graficar la ruta base con OSMnx (sin mostrar la ventana todavía)
    fig, ax = ox.plot_graph_route(
        G,
        ruta_nodos,
        route_color="#00FFFF",   # Cyan brillante (se ve excelente sobre satélite)
        route_linewidth=3.5,
        node_size=0,
        edge_color="white",
        edge_linewidth=0.3,
        show=False,
        close=False
    )
    # 2. URL directa al servidor de imágenes satelitales de ESRI
    esri_imagery_url = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    # 3. Superponer la capa satelital con Contextily
    try:
        cx.add_basemap(
            ax,
            crs=G.graph["crs"],       # Esencial: le indica a Contextily la proyección de OSMnx
            source=esri_imagery_url,  # Fuente directa a ESRI
            zoom=zoom_level           # Control de resolución manual (ej: 14, 15, 16)
        )
    except Exception as e:
        print(f"Atención: No se pudo cargar alguna tesela del mapa ({e}). Prueba bajando el nivel de zoom.")
    # 4. Ajustes finales de visualización
    ax.set_title("Ruta sobre Mapa Satelital", color="white", backgroundcolor="black", fontsize=12)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # OPCIÓN A: Descargar datos en vivo (Recomendado)
    # lugar = "Elche, Spain"
    # print(f"Descargando la red vial para: {lugar}...")
    # G = ox.graph_from_place(lugar, network_type="drive")
    # OPCIÓN B: Usar un archivo local descargado de Overpass Turbo
    print('Cargando el mapa...')
    G = ox.graph_from_xml("maps/map0.osm")
    print('Mapa cargado.')

    # EL MISMO ALGORITMO DE ANTES --> se debe modificar para manejar el nuevo mapa
    algoritmo = A_star_Algorithm_OSM(G)
    # Introduce unas coordenadas GPS (Latitud, Longitud) de origen y destino
    gps_origen = (38.2694, -0.706661)
    gps_destino = (38.25, -0.694)
    # Buscamos los nodos reales de la red más cercanos a nuestros puntos GPS
    nodo_origen = algoritmo.get_closest_node(gps_origen)
    nodo_destino = algoritmo.get_closest_node(gps_destino)
    print(f"ID Nodo Origen más cercano: {nodo_origen}")
    print(f"ID Nodo Destino más cercano: {nodo_destino}")
    route, distance, iterations = algoritmo.find_route(nodo_origen, nodo_destino)
    # Ahora ploteamos sobre un mapa real satelital. Ajusta el zoom entre 5 (grueso) y 16 (fino)
    plot_route(G, route, distance, zoom_level=5)
