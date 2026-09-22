import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import contextily as cx
import osmnx as ox



class A_star_Algorithm_OSM():
    """
    Modified to work with OSM
    """
    def __init__(self, graph):
        self.graph = graph
        self.queue = []
        self.visited_nodes = []
        self.node_info = {}
        # se inicializa g y f con
        self.g_scores = defaultdict(lambda: np.inf)
        self.f_scores = defaultdict(lambda: np.inf)

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
        # Extract lat (y) and lon (x) for both nodes
        lat1, lon1 = self.graph.nodes[current_node_name]['y'], self.graph.nodes[current_node_name]['x']
        lat2, lon2 = self.graph.nodes[destination_name]['y'], self.graph.nodes[destination_name]['x']
        # Calculate great-circle distance in meters
        distance_meters = ox.distance.great_circle(lat1, lon1, lat2, lon2)
        return distance_meters

    def process_neighbors_A_star(self, current_node_name, destination_name):
        # Get the list of neighbors of the current node
        neighbors = self.graph.neighbors(current_node_name)
        print('Found neighbors:', neighbors)
        # cumulative distance of current node
        for neighbor in neighbors:
            # Optional: check that you can travel from node A to B. If no edge exists, just skip it.
            # This is recommended, according to the osmnx library
            if not self.graph.has_edge(current_node_name, neighbor):
                continue
            # se calcula el valor acumulado, añadiendo el tramo de carretera
            datos_aristas = self.graph.get_edge_data(current_node_name, neighbor)
            d = datos_aristas[0].get('length')
            tentative_g_score = self.g_scores[current_node_name] + d
            if tentative_g_score < self.g_scores[neighbor]:
                self.node_info[neighbor] = {'parent': current_node_name}
                self.g_scores[neighbor] = tentative_g_score
                # h = self.graph.compute_flying_distance(neighbor, destination_name)
                h = self.compute_flying_distance(neighbor, destination_name)
                self.f_scores[neighbor] = tentative_g_score + h
                if neighbor not in self.queue:
                    self.queue.append(neighbor)

    def reorder_queue(self):
        """
        TODO: quitar función, debe añadirla el estudiante.
        Al añadir esta función, ya no es BFS, sino que en la cola se incluye el nodo con mayor interés
        (menor distancia). Al reordenarse, se procesará en la siguiente iteración este nodo.
        :param destination:
        :return:
        """
        distances = []
        for node_name in self.queue:
            f = self.g_scores[node_name]
            distances.append(f)
        sort_index = np.argsort(distances)
        # reorder queue according to current flying distances
        self.queue = [self.queue[i] for i in sort_index]

    def reconstruct_route(self, current_node):
        route = []
        # total_distance = 0# en este caso, el último nodo ya incluye la distancia acumulada
        total_distance = self.g_scores[current_node]
        while current_node is not None:
            route.append(current_node)
            current_node = self.node_info[current_node].get('parent')
        # Reverse to get start -> destination
        route = route[::-1]
        return route, total_distance

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
        self.queue = [start_name]
        self.node_info[start_name] = {'parent': None}
        self.g_scores[start_name] = 0
        # f(start) = g + h = h
        self.f_scores[start_name] = self.compute_flying_distance(start_name, destination_name) # g + h
        iterations = 0
        while len(self.queue) > 0:
            print(30 * '=')
            print('Current queue is: ', self.queue)
            # pop current_node from the queue
            current_node = self.queue.pop(0)
            print('Current node is: ', current_node)
            if current_node == destination_name:
                print('Found destination! In iterations: ', iterations)
                # Found solution: destination reached --> reconstruct the route
                route, distance = self.reconstruct_route(current_node)
                return route, distance, iterations
            self.process_neighbors_A_star(current_node, destination_name)
            self.reorder_queue()
            iterations += 1
        # No route exists
        return None, None, iterations


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


