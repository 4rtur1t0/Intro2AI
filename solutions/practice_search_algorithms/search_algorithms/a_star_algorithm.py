import numpy as np
from collections import defaultdict

class A_star_Algorithm():
    def __init__(self, graph):
        self.graph = graph
        self.open_queue = []
        self.closed_queue = []
        self.node_info = {}
        self.g_scores = defaultdict(lambda: np.inf)
        self.h_scores = defaultdict(lambda: np.inf)

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

    def get_next_node(self):
        # sort with respect to f = g + h and pop the minimum
        self.open_queue.sort(key=lambda x: self.g_scores[x] + self.h_scores[x])
        return self.open_queue.pop(0)

    def process_neighbors(self, current_node, destination_name):
        # Get the list of neighbors of the current node
        neighbors = self.graph.get_neighbors(current_node)
        # print('Found neighbors:', neighbors)
        # cumulative distance of current node
        for neighbor in neighbors:
            if neighbor in self.closed_queue:
                continue
            # se calcula el valor acumulado, añadiendo el tramo intermedio de carretera
            temp_dist = self.g_scores[current_node] + self.graph.get_distance(current_node, neighbor)
            if neighbor not in self.open_queue:
                self.open_queue.append(neighbor)
                self.g_scores[neighbor] = temp_dist
                self.h_scores[neighbor] = self.graph.compute_flying_distance(neighbor, destination_name)
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
        if not self.graph.get_node(start_name) or not self.graph.get_node(destination_name):
            return None, None, None
        self.g_scores[start_name] = 0
        self.h_scores[start_name] = self.graph.compute_flying_distance(start_name, destination_name)
        self.node_info[start_name] = {'parent': None}
        self.open_queue.append(start_name)
        route = None
        distance = None
        iterations = 0
        while len(self.open_queue) > 0:
            current_node = self.get_next_node()
            print(30 * '=')
            print('Current node is: ', current_node)
            print(30 * '=')
            self.closed_queue.append(current_node)
            if current_node == destination_name:
                route, distance = self.reconstruct_route(destination_name)
                break
            self.process_neighbors(current_node, destination_name)
        print('THE MINIMUM DISTANCES FROM START TO OTHER NODES:', start_name)
        print(self.g_scores)
        # print('THE RANKING DISTANCES:', start_name)
        # print(self.f_scores)
        return route, distance, iterations



