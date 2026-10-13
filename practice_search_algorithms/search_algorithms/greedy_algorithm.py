"""
Greedy Algorithm
"""
import numpy as np

class Greedy_Algorithm():
    def __init__(self, graph):
        self.graph = graph
        self.open_queue = []    # llista oberta
        self.closed_queue = []  # llista tancada
        self.node_info = {}
        self.h = {}

    def print_info(self, iteration, current_node):
        print(100*'#')
        print('ITERATION:', iteration)
        print(100 * '-')
        print('CURRENT NODE:', current_node)
        print('OPEN LIST')
        print(self.open_queue)
        print(100 * '-')
        print('CLOSED LIST')
        print(self.closed_queue)
        print(100 * '#')

    def reorder_queue(self):
        """
        Al añadir esta función, ya no es BFS, sino que en la cola se incluye el nodo con mayor interés
        (menor distancia). Al reordenarse, se procesará en la siguiente iteración este nodo.
        :param destination:
        :return:
        """
        ###########################################################################################
        # EJERCICIO 2.8: debe reordenarse la cola abierta (self.open_queue) en base a los valores de la heurística h
        ###########################################################################################
        return

    def reconstruct_route(self, current_node_name):
        """
        Troba la ruta calculada des d'el node final fins a l'inicial buscant en el diccionari node_info
        :param current_node:
        :return:
        """
        route = []
        total_distance = 0
        while current_node_name is not None:
            route.append(current_node_name)
            current_node_name = self.node_info[current_node_name].get('parent')
        # Reverse to get start -> destination
        route = route[::-1]
        for i in range(len(route) - 1):
            nodeA = route[i]
            nodeB = route[i + 1]
            d = self.graph.get_distance(nodeA, nodeB)
            total_distance += d
        return route, total_distance

    def process_neighbors(self, current_node, destination_node):
        """
        Obtain the neighbours (successors) of the current node.
            For each node:
                if the neighbor is not in the list of visited nodes:
                    append it to the list of visited nodes.
                    append it to the node processing queue (FIFO queue)
        Para cada vecino neighbour encontrado, se comprueba si ya se ha visitado. Si ya se ha visitado, se continúa.
        si no se ha visitado, se añade a la lista de nodos visitados, se añade al diccionario de información de nodos
        (node_info) y se añade a la cola
        :param current_node:
        :return:
        """
        # Get the list of neighbors of the current node
        neighbors = self.graph.get_neighbors(current_node)
        print('Found neighbors:', neighbors)
        for neighbor in neighbors:
            # si ja s'havia visitat
            if neighbor in self.closed_queue:
                continue
            # en cas contrari
            if neighbor not in self.open_queue:
                self.open_queue.append(neighbor)
                self.node_info[neighbor] = {'parent': current_node}
                ###########################################################################################
                # EJERCICIO 2.8: OJO! debemos calcular la distancia en línea recta hasta el destino
                ###########################################################################################

    def find_route(self, start_name, destination_name):
        """Finds a route using iterative Breadth-First Search."""
        # Safety check: ensure both cities actually exist in our graph
        if not self.graph.get_node(start_name) or not self.graph.get_node(destination_name):
            print('Either the start node or the destination node does not exist')
            return None, None, None
        # se inicializa la cola y la lista de nodos visitados
        self.open_queue = [start_name]
        self.closed_queue = []
        self.node_info[start_name] = {'parent': None}
        self.h[start_name] = self.graph.compute_flying_distance(start_name, destination_name)
        iteration = 0
        self.print_info(iteration=iteration, current_node=None)
        while len(self.open_queue) > 0:
            current_node = self.open_queue.pop(0)
            self.closed_queue.append(current_node)
            self.print_info(iteration=iteration, current_node=current_node)
            if current_node == destination_name:
                route, distance = self.reconstruct_route(current_node)
                return route, distance, iteration
            self.process_neighbors(current_node, destination_name)
            #########################################################
            # EJERCICIO 2.8: PROGRAME Y LLAME a self.reorder_queue para reordenar la cola
            # y poder elegir el nodo con menor h
            #########################################################
        return None, None, iteration  # No route exists

