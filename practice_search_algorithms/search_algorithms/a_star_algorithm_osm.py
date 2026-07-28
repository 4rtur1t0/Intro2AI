import numpy as np
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


    def process_neighbors(self, current_node_name, destination_name):
        """
        Obtain the neighbours (successors) of the current node.
            For each node:
                if the neighbor is not in the list of visited nodes:
                    append it to the list of visited nodes.
                    append it to the node processing queue (FIFO queue)

                    compute accumulated_distance
                    compute flying_distance
                    compute
        :param current_node:
        :return:
        """
        #
        # COMPLETE ESTA FUNCIÓN DE FORMA SIMILAR AL ALGORITMO A*. Para adaptarse al
        # GRAFO de la librería osmnx se indican abajo las funciones más importantes
        #
        # Para encontrar los vecinos de un nodo en el grafo de osmnx
        # self.graph.neighbors(current_node_name)

        # Esta función comprueba si se puede viajar desde un nodo al siguiente
        # self.graph.has_edge(current_node_name, neighbor)

        # Para encontrar los datos entre dos nodos (distancia, tipo de vía)
        #datos_aristas = self.graph.get_edge_data(current_node_name, neighbor)

        # Para calcular la distancia en línea recta: se puede usar el método de esta clase
        #flying_distance = self.compute_flying_distance(neighbor, destination_name)


    def reorder_queue(self):
        """
        Como en el algoritmo A*
        :return:
        """
        # COMPLETE ESTA FUNCIÓN DE FORMA SIMILAR AL ALGORITMO A*.
        return

    def get_route(self, current_node):
        """
        Como en el algoritmo A*
        :return:
        """
        # COMPLETE ESTA FUNCIÓN DE FORMA SIMILAR AL ALGORITMO A*.
        return


    def find_route(self, start_name, destination_name):
        # COMPLETE ESTA FUNCIÓN DE FORMA SIMILAR AL ALGORITMO A*.
        return


