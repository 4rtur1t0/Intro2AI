import numpy as np


class Greedy_Algorithm():
    def __init__(self, graph):
        self.graph = graph
        self.queue = []
        self.queue_fd = [] # flying distances in queue
        self.visited_nodes = []
        self.node_info = {}

    def process_neighbors(self, current_node, destination_node):
        """
        Obtain the neighbours (successors) of the current node.
            For each node:
                if the neighbor is not in the list of visited nodes:
                    append it to the list of visited nodes.
                    append it to the node processing queue (FIFO queue)
        :param current_node:
        :return:
        """
        # Get the list of neighbors of the current node
        neighbors = self.graph.get_neighbors(current_node)
        print('Found neighbors:', neighbors)
        # para cada vecino neighbour encontrado
        for neighbor in neighbors:
            if neighbor not in self.visited_nodes:
                # si no se ha visitado: a) se añade a la lista de visitados y b) se añade a la cola de exploración
                self.visited_nodes.append(neighbor)
                flying_distance = self.graph.compute_flying_distance(neighbor, destination_node)
                self.node_info[neighbor] = {'parent': current_node}
                # node = {'name': neighbor, 'flying_distance': flying_distance}
                self.queue.append(neighbor)
                self.queue_fd.append(flying_distance)

    def reorder_queue(self):
        """
        TODO: quitar función, debe añadirla el estudiante.
        Al añadir esta función, ya no es BFS, sino que en la cola se incluye el nodo con mayor interés
        (menor distancia). Al reordenarse, se procesará en la siguiente iteración este nodo.
        :param destination:
        :return:
        """
        sort_index = np.argsort(self.queue_fd)
        # reorder the queue according to current flying distances
        self.queue = [self.queue[i] for i in sort_index]
        self.queue_fd = [self.queue_fd[i] for i in sort_index]

    def reconstruct_route(self, current_node):
        route = []
        total_distance = 0
        while current_node is not None:
            route.append(current_node)
            current_node = self.node_info[current_node].get('parent')
        # Reverse to get start -> destination
        route = route[::-1]
        # TODO: NOW COMPUTE the total distance of the solution
        # TODO: CALCULE LA DISTANCIA TOTAL
        for i in range(len(route) - 1):
            nodeA = route[i]
            nodeB = route[i + 1]
            d = self.graph.get_distance(nodeA, nodeB)
            total_distance += d
        return route, total_distance


    def find_route(self, start_name, destination_name):
        """Finds a route using iterative Breadth-First Search."""
        # Safety check: ensure both cities actually exist in our graph
        if not self.graph.get_node(start_name) or not self.graph.get_node(destination_name):
            return None
        # Standard BFS setup + starting flying distance
        self.queue = [start_name]
        self.queue_fd = [self.graph.compute_flying_distance(start_name, destination_name)]
        self.visited_nodes = [start_name]
        self.node_info[start_name] = {'parent': None}
        iterations = 0
        while len(self.queue) > 0:
            iterations += 1
            print(30 * '=')
            print('Current queue is: ', self.queue)
            # pop current_node from the queue
            current_node = self.queue.pop(0)
            # also pop its corresponding distance
            _ = self.queue_fd.pop(0)
            print('Current node is: ', current_node)
            if current_node == destination_name:
                print('Found destination! in iterations: ', iterations)
                # Found solution: destination reached --> reconstruct the route
                route, distance = self.reconstruct_route(current_node)
                return route, distance, iterations
            self.process_neighbors(current_node, destination_name)
            # TODO: CUIDADO! DEBEMOS REORDENAR LA LISTA DE ACUERDO CON NUESTRA HEURÍSTICA
            self.reorder_queue()
        return None, None, iterations  # No route exists
