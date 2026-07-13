import heapq
import osmnx as ox
import networkx as nx

"""
Navigate to 
overpass-turbo.eu
execute this query.
Use export --> raw data OSM
MAP 1: Detailed map, valid to download a city
[out:xml];
(
  way["highway"]({{bbox}});
);
out body;
>;
out skel qt;


MAP 2: larger area, less detail.
[out:xml][timeout:90];

// 1. Buscamos las vías principales dentro del mapa visible ({{bbox}})
(
  way["highway"~"^(motorway|trunk|primary|secondary)(_link)?$"]({{bbox}});
);

// 2. Devolvemos la información de estas calles
out body;

// 3. El operador ">" (recurse down) busca y descarga todos los nodos GPS
// que componen físicamente esas calles para que OSMnx pueda unirlos.
>;
out skel qt;
"""

# --- 1. IMPLEMENTACIÓN DEL ALGORITMO DE DIJKSTRA ---
def dijkstra_personalizado(G, inicio, destino):
    """
    Ejecuta el algoritmo de Dijkstra sobre un grafo de OSMnx (MultiDiGraph).
    Retorna la lista de IDs de nodos del camino más corto y la distancia total en metros.
    """
    # Inicializamos todas las distancias en infinito y los predecesores en None
    distancias = {nodo: float('inf') for nodo in G.nodes}
    distancias[inicio] = 0
    predecesores = {nodo: None for nodo in G.nodes}

    # Cola de prioridad: almacena tuplas de (distancia_acumulada, id_nodo)
    cola_prioridad = [(0, inicio)]

    while cola_prioridad:
        distancia_actual, nodo_actual = heapq.heappop(cola_prioridad)

        # Si ya alcanzamos el nodo de destino, podemos finalizar la búsqueda
        if nodo_actual == destino:
            break

        # Si encontramos un camino registrado más corto hacia este nodo, ignoramos el actual
        if distancia_actual > distancias[nodo_actual]:
            continue

        # Exploramos los vecinos del nodo actual
        for vecino in G.neighbors(nodo_actual):
            # Como OSMnx genera un MultiDiGraph, puede haber múltiples conexiones (aristas) entre dos nodos.
            # Nos quedamos con la arista que tenga la menor longitud ("length" en metros).
            datos_aristas = G.get_edge_data(nodo_actual, vecino)
            peso_arista = min(arista['length'] for arista in datos_aristas.values())

            nueva_distancia = distancia_actual + peso_arista

            if nueva_distancia < distancias[vecino]:
                distancias[vecino] = nueva_distancia
                predecesores[vecino] = nodo_actual
                heapq.heappush(cola_prioridad, (nueva_distancia, vecino))

    # Reconstrucción del camino desde el destino hacia el origen
    camino = []
    nodo_aux = destino
    while nodo_aux is not None:
        camino.append(nodo_aux)
        nodo_aux = predecesores[nodo_aux]

    camino.reverse()

    # Si el primer nodo reconstruido es el de inicio, la ruta es válida
    if camino[0] == inicio:
        return camino, distancias[destino]
    else:
        return None, float('inf')


# --- 2. OBTENER LOS DATOS DESDE OPENSTREETMAP ---

# OPCIÓN A: Descargar datos en vivo (Recomendado)
# OSMnx hará internamente la llamada a la API de Overpass por ti.
#lugar = "Toledo, Spain"
#print(f"Descargando la red vial para: {lugar}...")
#G = ox.graph_from_place(lugar, network_type="drive")

# OPCIÓN B: Usar un archivo local descargado de Overpass Turbo
# Si fuiste a overpass-turbo.eu y exportaste como "OSM XML", descomenta la siguiente línea:
print('Cargando el mapa...')
G = ox.graph_from_xml("maps/map2.osm")
print('Mapa cargado.')

# --- 3. DEFINIR COORDENADAS GPS DE ORIGEN Y DESTINO ---
# Ingresa las coordenadas reales (Latitud, Longitud)
gps_origen = (38.27408, -0.68406)  # Ejemplo: Cerca del Alcázar de Toledo
gps_destino = (45.4, -0.58)  # Ejemplo: Cerca del Puente de San Martín

# OSMnx requiere las coordenadas en formato (Longitud, Latitud) -> (X, Y)
# Buscamos los nodos reales de la red más cercanos a nuestros puntos GPS
nodo_origen = ox.distance.nearest_nodes(G, X=gps_origen[1], Y=gps_origen[0])
nodo_destino = ox.distance.nearest_nodes(G, X=gps_destino[1], Y=gps_destino[0])

print(f"ID Nodo Origen más cercano: {nodo_origen}")
print(f"ID Nodo Destino más cercano: {nodo_destino}")

# --- 4. EJECUTAR DIJKSTRA Y MOSTRAR RESULTADOS ---
print("Calculando la ruta más corta...")
ruta_nodos, distancia_total = dijkstra_personalizado(G, nodo_origen, nodo_destino)

if ruta_nodos:
    print("\n" + "=" * 30)
    print("¡Ruta calculada con éxito!")
    print(f"Distancia total: {distancia_total:.2f} metros (aprox. {distancia_total / 1000:.2f} km)")
    print(f"Número de nodos en la ruta: {len(ruta_nodos)}")
    print("=" * 30)

    # --- 5. GRAFICAR LA RUTA EN EL MAPA ---
    print("Dibujando el mapa...")
    ox.plot_graph_route(
        G,
        ruta_nodos,
        route_color="red",
        route_linewidth=4,
        node_size=0,
        edge_color="#999999"
    )
else:
    print("No fue posible encontrar una ruta transitable entre los dos puntos GPS.")