import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import contextily as cx
import osmnx as ox

class A_star_Algorithm_OSM():
    def __init__(self, graph):
        self.graph = graph
        self.open_queue = []
        self.closed_queue = []
        self.node_info = {}
        self.g_scores = defaultdict(lambda: np.inf)
        self.h_scores = defaultdict(lambda: np.inf)

    def get_node(self, node_id):
        return self.graph.nodes[node_id]

    def get_closest_node(self, gps):
        """
        Returns the node with GPS coordinates closest to the given ones.
        :param gps:
        :return:
        """
        # OSMnx requiere las coordenadas en formato (Longitud, Latitud) -> (X, Y)
        # Buscamos los nodos reales de la red más cercanos a nuestros puntos GPS
        node = ox.distance.nearest_nodes(self.graph, X=gps[1], Y=gps[0])
        return node

    def compute_flying_distance(self, current_node_name, destination_name):
        ################################################################
        # EJERCICIO 4.1: se debe completar el código usando las funciones
        # de osmnx.!:
        # ################################################################
        # Extract lat (y) and lon (x) for both nodes
        lat1, lon1 = self.graph.nodes[current_node_name]['y'], self.graph.nodes[current_node_name]['x']
        lat2, lon2 = self.graph.nodes[destination_name]['y'], self.graph.nodes[destination_name]['x']
        # Calculate great-circle distance in meters
        distance_meters = ox.distance.great_circle(lat1, lon1, lat2, lon2)
        return distance_meters

    def get_distance(self, current_node, neighbor):
        ################################################################
        # EJERCICIO 4.1: se debe completar el código usando las funciones
        # de osmnx.!:
        # ################################################################
        #if not self.graph.has_edge(current_node_name, neighbor):
        #    continue
        # se calcula el valor acumulado, añadiendo el tramo de carretera
        datos_aristas = self.graph.get_edge_data(current_node, neighbor)
        d = datos_aristas[0].get('length')
        return d

    def get_neighbors(self, current_node):
        ################################################################
        # EJERCICIO 4.1: se debe completar el código usando las funciones
        # de osmnx.!:
        # ################################################################
        return self.graph.neighbors(current_node)

    def reconstruct_route(self, current_node):
        route = []
        # en este caso, el último nodo ya incluye la distancia acumulada
        total_distance = self.g_scores[current_node]
        while current_node is not None:
            route.append(current_node)
            current_node = self.node_info[current_node].get('parent')
        # Reverse to get start -> destination
        route = route[::-1]
        return route, total_distance

    def get_next_node(self):
        # sort with respect to f = g + h and pop the minimum
        self.open_queue.sort(key=lambda x: self.g_scores[x] + self.h_scores[x])
        return self.open_queue.pop(0)

    def process_neighbors(self, current_node, destination_name):
        # Get the list of neighbors of the current node
        neighbors = self.get_neighbors(current_node)
        for neighbor in neighbors:
            if neighbor in self.closed_queue:
                continue
            # se calcula el valor acumulado, añadiendo el tramo intermedio de carretera
            ###############################################################
            # EJERCICIO 4.1! self.graph.distance --> self.get_distance
            # complete el código
            ###############################################################
            temp_dist = self.g_scores[current_node] + self.get_distance(current_node, neighbor)
            if neighbor not in self.open_queue:
                self.open_queue.append(neighbor)
                self.g_scores[neighbor] = temp_dist
                ###############################################################
                # EJERCICIO 4.1: self.graph.compute_flying_distance --> self.compute_flying_distance
                # complete el código
                ###############################################################
                self.h_scores[neighbor] = self.compute_flying_distance(neighbor, destination_name)
                self.node_info[neighbor] = {'parent': current_node}
            if temp_dist < self.g_scores[neighbor]:
                self.g_scores[neighbor] = temp_dist
                self.node_info[neighbor] = {'parent': current_node}

    def find_route(self, start_name, destination_name):
        """
        Una implementación de A star
        :param start_name:
        :param destination_name:
        :return:
        """
        # Safety check: ensure both cities actually exist in our graph
        if start_name not in self.graph or destination_name not in self.graph:
            print("No se encuentra el nodo de origen o el de destino en el grafo")
            return None, None, None
        self.g_scores[start_name] = 0
        ###############################################################
        # EJERCICIO 4.1!: CUIDADO, se debe definir compute_flying_distance función con el grafo de OSM
        # complete el código
        ###############################################################
        self.h_scores[start_name] = self.compute_flying_distance(start_name, destination_name)
        self.node_info[start_name] = {'parent': None}
        self.open_queue.append(start_name)
        route = None
        distance = None
        iterations = 0
        while len(self.open_queue) > 0:
            iterations += 1
            current_node = self.get_next_node()
            print(30 * '=')
            print('Current node is: ', current_node)
            print(30 * '=')
            self.closed_queue.append(current_node)
            if current_node == destination_name:
                route, distance = self.reconstruct_route(destination_name)
                break
            self.process_neighbors(current_node, destination_name)
        #print('THE MINIMUM DISTANCES FROM START TO OTHER NODES:', start_name)
        #print(self.g_scores)
        return route, distance, iterations


    def plot_route(self, ruta_nodos, distancia_total, zoom_level=15):
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
            self.graph,
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
                crs=self.graph.graph["crs"],    # Esencial: le indica a Contextily la proyección de OSMnx
                source=esri_imagery_url,  # Fuente directa a ESRI
                zoom=zoom_level           # Control de resolución manual (ej: 14, 15, 16)
                )
        except Exception as e:
            print(f"Atención: No se pudo cargar alguna tesela del mapa ({e}). Prueba bajando el nivel de zoom.")
        # 4. Ajustes finales de visualización
        ax.set_title("Ruta sobre el mapa satelital", fontsize=12)
        plt.tight_layout()
        plt.show()
