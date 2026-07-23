"""
BFS Algorithm
"""
class BFS_Algorithm():
    def __init__(self, graph):
        self.graph = graph
        self.queue = []
        self.visited_nodes = []
        # saves the parent of each node to backtrack the route
        self.node_info = {}

    def process_neighbors(self, current_node_name):
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
        neighbors = self.graph.get_neighbors(current_node_name)
        print('Found neighbors:', neighbors)
        # para cada vecino neighbour encontrado, se comprueba si ya se ha visitado. Si ya se ha visitado, se continúa.
        # si no se ha visitado, se añade a la lista de nodos visitados, se añade al diccionario de información de nodos
        # (node_info) y se añade a la cola
        for neighbor in neighbors:
            if neighbor not in self.visited_nodes:
                # si no se ha visitado: a) se añade a la lista de visitados y b) se añade a la cola de exploración
                self.visited_nodes.append(neighbor)
                self.node_info[neighbor] = {'parent': current_node_name}
                self.queue.append({'name': neighbor})

    def get_route(self, current_node_name):
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
        # se inicializa la cola y la lista de nodos visitados
        self.queue = [{'name': start_name}]
        self.visited_nodes = [start_name]
        # el node inicial no té pare
        self.node_info[start_name] = {'parent': None}
        iterations = 0
        while len(self.queue) > 0:
            iterations += 1
            print('Current queue is: ', self.queue)
            # pop the current_node from the queue
            current_node = self.queue.pop(0)
            current_node_name = current_node.get('name')
            print('Current node is: ', current_node)
            if current_node_name == destination_name:
                print('Found destination! in iterations: ', iterations)
                route, distance = self.get_route(current_node_name)
                return route, distance, iterations
            self.process_neighbors(current_node_name)
        return None, None, iterations  # No route exists

