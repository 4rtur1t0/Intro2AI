"""
BFS Algorithm
"""
class BFS_Algorithm():
    def __init__(self, graph):
        self.graph = graph
        self.open_queue = []    # llista oberta
        self.closed_queue = []  # llista tancada
        # saves the parent of each node to backtrack the route
        self.node_info = {}

    def print_info(self, iteration, current_node):
        # TODO: print info!
        return

    def reconstruct_route(self, current_node_name):
        """
        Troba la ruta calculada des d'el node final fins a l'inicial buscant en el diccionari node_info
        :param current_node:
        :return:
        """
        route = []
        total_distance = 0
        # TODO: DEBE DEVOLVER la ruta calculada y la distancia total
        return route, total_distance

    def process_neighbors(self, current_node):
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
        # print('Found neighbors:', neighbors)
        for neighbor in neighbors:
            # si el node ja ha sigut explorat amb anterioritat
            if neighbor in self.closed_queue:
                continue
            # si no es troba a la llista oberta, s'ha d'afegir
            if neighbor not in self.open_queue:
                self.open_queue.append(neighbor)
                self.node_info[neighbor] = {'parent': current_node}

    def find_route(self, start_name, destination_name):
        """Finds a route using iterative Breadth-First Search."""
        # Safety check: ensure both cities actually exist in our graph
        if not self.graph.get_node(start_name) or not self.graph.get_node(destination_name):
            print('Either the start node or the destination node does not exist')
            return None, None, None
        # se inicializa la cola y la lista de nodos visitados
        self.open_queue = [start_name]
        self.closed_queue = []
        # el node inicial no té pare
        self.node_info[start_name] = {'parent': None}
        iteration = 0
        # self.print_info(iteration=iteration, current_node=None)
        while len(self.open_queue) > 0:
            iteration += 1
            current_node = self.open_queue.pop(0)
            # add the current_node to the closed queue
            self.closed_queue.append(current_node)
            # self.print_info(iteration=iteration, current_node=current_node)
            if current_node == destination_name:
                route, distance = self.reconstruct_route(current_node)
                return route, distance, iteration
            self.process_neighbors(current_node)
        return None, None, iteration  # No route exists

