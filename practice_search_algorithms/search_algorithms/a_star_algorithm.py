import numpy as np
from collections import defaultdict

class A_star_AlgorithmB():
    def __init__(self, graph):
        self.graph = graph
        self.queue = []
        self.visited_nodes = []
        self.node_info = {}
        # se inicializa g y f con
        self.g_scores = defaultdict(lambda: np.inf)
        self.f_scores = defaultdict(lambda: np.inf)

    def process_neighbors_A_star(self, current_node_name, destination_name):
        # Get the list of neighbors of the current node
        neighbors = self.graph.get_neighbors(current_node_name)
        print('Found neighbors:', neighbors)
        # cumulative distance of current node
        for neighbor in neighbors:
            # se calcula el valor acumulado, añadiendo el tramo de carretera
            d = self.graph.get_distance(current_node_name, neighbor)
            tentative_g_score = self.g_scores[current_node_name] + d
            if tentative_g_score < self.g_scores[neighbor]:
                self.node_info[neighbor] = {'parent': current_node_name}
                self.g_scores[neighbor] = tentative_g_score
                h = self.graph.compute_flying_distance(neighbor, destination_name)
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
        if not self.graph.get_node(start_name) or not self.graph.get_node(destination_name):
            return None
        self.queue = [start_name]
        self.node_info[start_name] = {'parent': None}
        self.g_scores[start_name] = 0
        # f(start) = g + h = h
        self.f_scores[start_name] = self.graph.compute_flying_distance(start_name, destination_name) # g + h
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
        # No route exists
        return None, None, iterations


