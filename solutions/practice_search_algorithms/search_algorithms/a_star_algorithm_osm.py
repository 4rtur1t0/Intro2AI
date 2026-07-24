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


    def process_neighbors_A_star(self, current_node_name, destination_name):
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
        # Get the list of neighbors of the current node
        # neighbors = self.graph.neighbors(current_node_name)
        # print('Found neighbors:', neighbors)
        # cumulative distance of current node
        current_accumulated_distance = self.node_info[current_node_name].get('accumulated_distance')
        # para cada vecino neighbour encontrado
        for neighbor in self.graph.neighbors(current_node_name):
            # check that you can travel from node A to B
            # if no edge exists, just skip it
            if not self.graph.has_edge(current_node_name, neighbor):
                continue
            if neighbor not in self.visited_nodes:
                # rel distance to travel to the neighbor from current node
                datos_aristas = self.graph.get_edge_data(current_node_name, neighbor)
                rel_distance = datos_aristas[0].get('length')
                # if speed is needed: if datos_aristas[0].get('maxspeed'):
                new_accumulated_distance = current_accumulated_distance + rel_distance
                flying_distance = self.compute_flying_distance(neighbor, destination_name)
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
        if start_name not in self.graph or destination_name not in self.graph:
            print("No se encuentra el nodo de origen o el de destino en el grafo")
        # if not self.graph.get_node(start_name) or not self.graph.get_node(destination_name):
            return None, None, None
        # Standard BFS setup
        self.queue = [{'name': start_name, 'ranking_distance': 0}]
        self.visited_nodes = [start_name]
        self.node_info[start_name] = {'parent': None, 'accumulated_distance': 0}

        iterations = 0
        while len(self.queue) > 0:
            iterations += 1
            # print(30 * '=')
            # print('Current queue is: ', self.queue)
            # pop current_node from the queue
            current_node = self.queue.pop(0)
            current_node_name = current_node['name']
            # print('Current node is: ', current_node)
            if current_node_name == destination_name:
                print('Found destination! in iterations: ', iterations)
                # Found solution: destination reached --> reconstruct the route
                route, distance = self.get_route(current_node_name)
                return route, distance, iterations
            self.process_neighbors_A_star(current_node_name, destination_name)
            # TODO: CUIDADO! DEBEMOS REORDENAR LA LISTA DE ACUERDO CON NUESTRA HEURÍSTICA
            self.reorder_queue()
        return None, None, iterations  # No route exists


