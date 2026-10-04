import numpy as np
from collections import defaultdict

class Dijkstra_Algorithm():
    def __init__(self, graph):
        self.graph = graph
        self.marked_nodes = []
        self.node_info = {}
        self.distances = defaultdict(lambda: np.inf)

    def reconstruct_route(self, current_node):
        route = []
        # total_distance = 0# en este caso, el último nodo ya incluye la distancia acumulada
        total_distance = self.distances[current_node]
        while current_node is not None:
            route.append(current_node)
            current_node = self.node_info[current_node].get('parent')
        # Reverse to get start -> destination
        route = route[::-1]
        return route, total_distance

    def get_next_node(self):
        # get a list of non-visited nodes and find the min distance
        dists = []
        names = []
        for k, v in self.distances.items():
            if v==float('inf'):
                continue
            if k not in self.marked_nodes:
                dists.append(v)
                names.append(k)
        current_node = names[np.argmin(dists)]
        return current_node

    def process_neighbors(self, current_node):
        # Get the list of neighbors of the current node
        neighbors = self.graph.get_neighbors(current_node)
        print('Found neighbors:', neighbors)
        # cumulative distance of current node
        for neighbor in neighbors:
            if neighbor in self.marked_nodes:
                continue
            # se calcula el valor acumulado, añadiendo el tramo de carretera
            d = self.graph.get_distance(current_node, neighbor)
            temp_dist = self.distances[current_node] + d
            if temp_dist < self.distances[neighbor]:
                self.distances[neighbor] = temp_dist
                self.node_info[neighbor] = {'parent': current_node}

    def find_route(self, start_name, destination_name):
        """
        Una implementación de A star
        :param start_name:
        :param destination_name:
        :return:
        """
        if not self.graph.get_node(start_name) or not self.graph.get_node(destination_name):
            return None
        for node in self.graph.get_nodes():
            self.distances[node] = float('inf')
        self.distances[start_name] = 0
        self.node_info[start_name] = {'parent': None}
        iterations = 0
        while True:
            print(30 * '=')
            current_node = self.get_next_node()
            print('Current node is: ', current_node)
            self.marked_nodes.append(current_node)
            # si queremos encontrar una única distancia a un nodo
            #if current_node == destination_name:
            #    break
            # Si queremos encontrar la mínima distancia (y ruta) a todos
            # los nodos del mapa
            if len(self.marked_nodes)==len(self.graph.get_nodes()):
                break
            self.process_neighbors(current_node)
        route, distance = self.reconstruct_route(destination_name)
        print('THE MINIMUM DISTANCES TO ALL GRAPHS STARTING FROM:', start_name)
        print(self.distances)
        return route, distance, iterations



