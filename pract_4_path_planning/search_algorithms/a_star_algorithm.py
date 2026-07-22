import numpy as np

class A_star_Algorithm():
    def __init__(self, graph):
        self.graph = graph
        self.queue = []
        self.visited_nodes = []
        self.node_info = {}

    def process_neighbors_A_star(self, current_node_name, destination_name):
        """
        Obtain the neighbours (successors) of the current node.
            For each node:
                if the neighbor is not in the list of visited nodes:
                    append it to the list of visited nodes.
                    append it to the node processing queue (FIFO queue)

                    compute cum_distance
                    compute flying_distance
        :param current_node:
        :return:
        """
        # Get the list of neighbors of the current node
        neighbors = self.graph.get_neighbors(current_node_name)
        print('Found neighbors:', neighbors)
        # cumulative distance of current node
        current_accumulated_distance = self.node_info[current_node_name].get('accumulated_distance')
        # para cada vecino neighbour encontrado
        for neighbor in neighbors:
            if neighbor not in self.visited_nodes:
                # rel distance to travel to the neighbor from current node
                rel_distance = self.graph.get_distance(current_node_name, neighbor)
                new_accumulated_distance = current_accumulated_distance + rel_distance
                flying_distance = self.graph.compute_flying_distance(neighbor, destination_name)
                # si no se ha visitado: a) se añade a la lista de visitados y b) se añade a la cola de exploración
                # print('Adding new node to the queue:', neighbor)
                self.visited_nodes.append(neighbor)
                # TODO: QUITAR parent_map --> debe guardar la ruta el alumno
                self.node_info[neighbor] = {'parent': current_node_name, 'accumulated_distance': new_accumulated_distance}
                self.queue.append({'name': neighbor, 'ranking_distance': new_accumulated_distance + flying_distance})

    def reorder_queue(self):
        """
        TODO: quitar función, debe añadirla el estudiante.
        Al añadir esta función, ya no es BFS, sino que en la cola se incluye el nodo con mayor interés
        (menor distancia). Al reordenarse, se procesará en la siguiente iteración este nodo.
        :param destination:
        :return:
        """
        distances = []
        for node in self.queue:
            d = node.get('ranking_distance')
            distances.append(d)
        sort_index = np.argsort(distances)
        # reorder queue according to current flying distances
        self.queue = [self.queue[i] for i in sort_index]

    def get_route(self, current_node):
        route = []
        # total_distance = 0# en este caso, el último nodo ya incluye la distancia acumulada
        total_distance = self.node_info[current_node].get('accumulated_distance')
        while current_node is not None:
            route.append(current_node)
            current_node = self.node_info[current_node].get('parent')
        # Reverse to get start -> destination
        route = route[::-1]
        return route, total_distance



    def find_route(self, start_name, destination_name):
        """Finds a route using iterative Breadth-First Search."""
        # Safety check: ensure both cities actually exist in our graph
        if not self.graph.get_node(start_name) or not self.graph.get_node(destination_name):
            return None
        # Standard BFS setup
        self.queue = [{'name': start_name, 'ranking_distance': 0}]
        self.visited_nodes = [start_name]
        self.node_info[start_name] = {'parent': None, 'accumulated_distance': 0}

        iterations = 0
        while len(self.queue) > 0:
            iterations += 1
            print(30 * '=')
            print('Current queue is: ', self.queue)
            # pop current_node from the queue
            current_node = self.queue.pop(0)
            current_node_name = current_node['name']
            print('Current node is: ', current_node)
            if current_node_name == destination_name:
                print('Found destination! in iterations: ', iterations)
                # Found solution: destination reached --> reconstruct the route
                route, distance = self.get_route(current_node_name)
                return route, distance, iterations
            self.process_neighbors_A_star(current_node_name, destination_name)
            # TODO: CUIDADO! DEBEMOS REORDENAR LA LISTA DE ACUERDO CON NUESTRA HEURÍSTICA
            self.reorder_queue()
        return None, None, iterations  # No route exists


    # BFS_search1
    # se entiende el mapa
    # se entiende cómo se van añadiendo nodos al mapa
    # PROBLEMA: se añaden bucles infinitos Madrid --> Segovia; Segovia--> Madrid
    # Se consigue saber si una ciudad y otra del mapa están conectadas (lo que no es demasiado útil,
    # ya que en principio están todos los nodos conectados de alguna manera
    # se propone al estudiante que calcule:
    #   a) el número de saltos entre Madrid y Elche
    #   b) se propone al estudiante que cree una lista de nodos visitados de forma absoluta, de esta manera
    #       se evita visitar nodos de forma infinita
    #   c) se debe guardar el predecesor de cada nodo, de manera que podamos guardar la ruta

    # BFS_search2
    # sin bucles infinitos
    # guarda la ruta
    # la ruta no es óptima en términos de distancia

    # Dijkstra

    # A*

    # Comparación
