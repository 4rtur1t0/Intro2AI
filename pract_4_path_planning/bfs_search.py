import numpy as np
import math
import matplotlib.pyplot as plt


class Graph:
    """Manages the collection of nodes and network algorithms."""

    def __init__(self):
        # Master dictionary to look up Node objects by their string name
        # Format: {"Madrid": <Node Object>}
        self.graph = {}
        self.gps_positions = None
        self.gps_position_data()

    # Function to add connections (undirected roads)
    def add_edge(self, node1_name, node2_name):
        # Ensure city1 exists in the graph
        if node1_name not in self.graph:
            self.graph[node1_name] = {}
        # Ensure city2 exists in the graph
        if node2_name not in self.graph:
            self.graph[node2_name] = {}
        distance = self.road_distance_data(node1_name, node2_name)
        # Connect both cities (two-way road)
        self.graph[node1_name][node2_name] = distance
        self.graph[node2_name][node1_name] = distance

    def get_node(self, name):
        """Safely fetches a Node object by name, returns None if missing."""
        return self.graph.get(name)

    def get_distance(self, node1, node2):
        """Safely fetches the distance between two nodes"""
        if node1 in self.graph and node2 in self.graph:
            return self.graph[node1][node2]
        else:
            return None

    def get_neighbors(self, name):
        neighbors = self.graph.get(name)
        neighbors = list(neighbors.keys())
        return neighbors

    def build_network(self):
        # 3. Build your city network
        self.add_edge("Madrid", "Guadalajara")
        self.add_edge("Madrid", "Cuenca")
        self.add_edge("Madrid", "Segovia")
        self.add_edge("Madrid", "Toledo")
        # self.add_edge("Madrid", "Albacete")
        self.add_edge("Madrid", "Ciudad Real")
        self.add_edge("Madrid", "Tarancón")
        self.add_edge("Tarancón", "Albacete")
        self.add_edge("Ciudad Real", "Puertollano")
        self.add_edge("Guadalajara", "Calatayud")
        self.add_edge("Calatayud", "Soria")
        self.add_edge("Cuenca", "Albacete")
        self.add_edge("Cuenca", "Valencia")
        self.add_edge("Valencia", "Gandia")
        self.add_edge("Gandia", "Alicante")
        self.add_edge("Albacete", "Villena")
        self.add_edge("Villena", "Alicante")
        self.add_edge("Alicante", "Elche")
        self.add_edge("Murcia", "Elche")
        self.add_edge("Murcia", "Albacete")

    def gps_position_data(self):
        self.gps_positions = {
            "Madrid": {"lat": 40.4167, "lon": -3.7037},
            "Guadalajara": {"lat": 40.6337, "lon": -3.1674},
            "Cuenca": {"lat": 40.0704, "lon": -2.1374},
            "Teruel": {"lat": 40.3456, "lon": -1.1065},
            "Segovia": {"lat": 40.9429, "lon": -4.1088},
            "Toledo": {"lat": 39.8567, "lon": -4.0244},
            "Soria": {"lat": 41.7666, "lon": -2.4667},
            "Ciudad Real": {"lat": 38.9848, "lon": -3.9274},
            "Puertollano": {"lat": 38.6833, "lon": -4.1167},
            "Calatayud": {"lat": 41.3532, "lon": -1.6468},
            "Zaragoza": {"lat": 41.6488, "lon": -0.8891},
            "Albacete": {"lat": 38.9943, "lon": -1.8585},
            "Villena": {"lat": 38.6318, "lon": -0.8612},
            "Alicante": {"lat": 38.3452, "lon": -0.4810},
            "Elche": {"lat": 38.2699, "lon": -0.7126},
            "Orihuela": {"lat": 38.0848, "lon": -0.9440},
            "Murcia": {"lat": 37.9922, "lon": -1.1307},
            "Valencia": {"lat": 39.4699, "lon": -0.3763},
            "Gandia": {"lat": 38.9680, "lon": -0.1830},
            "Tarancón": {"lat": 40.008279, "lon": -2.9958}}

    def road_distance_data(self, origen, destino):
        # Ciudades en orden:
        # 0: Madrid, 1: Guadalajara, 2: Cuenca, 3: Teruel, 4: Segovia, 5: Toledo, 6: Soria,
        # 7: Ciudad Real, 8: Puertollano, 9: Calatayud, 10: Zaragoza, 11: Albacete,
        # 12: Villena, 13: Alicante, 14: Elche, 15: Orihuela, 16: Murcia, 17: Valencia, 18: Gandia
        ciudades = ["Madrid", "Guadalajara", "Cuenca", "Teruel", "Segovia", "Toledo", "Soria",
                    "Ciudad Real", "Puertollano", "Calatayud", "Zaragoza", "Albacete",
                    "Villena", "Alicante", "Elche", "Orihuela", "Murcia", "Valencia", "Gandia", "Tarancón"]
        idx_orig = ciudades.index(origen)
        idx_dest = ciudades.index(destino)
        # Matriz de distancias por carretera en kilómetros (20x20)
        distancias_carretera = [
            # Madrid, Guad, Cuenc, Teruel, Segov, Toled, Soria, CdReal, Puer, Calat, Zgza, Albac, Vill, Alic, Elche, Orih, Murc, Valen, Gand, Taran
            [0, 60, 165, 300, 90, 72, 225, 210, 240, 235, 315, 250, 355, 420, 415, 425, 400, 355, 410, 85],  # Madrid
            [60, 0, 135, 240, 145, 130, 170, 265, 295, 175, 255, 255, 370, 435, 430, 440, 420, 360, 415, 105],
            # Guadalajara
            [165, 135, 0, 148, 255, 175, 245, 225, 260, 200, 290, 135, 235, 300, 295, 290, 280, 200, 230, 82],  # Cuenca
            [300, 240, 148, 0, 385, 335, 220, 370, 405, 135, 170, 280, 285, 305, 310, 325, 330, 140, 205, 230],
            # Teruel
            [90, 145, 255, 385, 0, 160, 185, 300, 330, 320, 400, 340, 445, 510, 505, 515, 490, 445, 500, 175],
            # Segovia
            [72, 130, 175, 335, 160, 0, 295, 120, 155, 305, 385, 220, 330, 395, 385, 385, 360, 335, 380, 100],  # Toledo
            [225, 170, 245, 220, 185, 295, 0, 430, 460, 95, 160, 380, 485, 550, 545, 555, 535, 360, 425, 250],  # Soria
            [210, 265, 225, 370, 300, 120, 430, 0, 38, 420, 500, 220, 315, 380, 365, 355, 335, 350, 370, 170],
            # Ciudad Real
            [240, 295, 260, 405, 330, 155, 460, 38, 0, 455, 535, 255, 350, 415, 400, 390, 370, 385, 405, 205],
            # Puertollano
            [235, 175, 200, 135, 320, 305, 95, 420, 455, 0, 88, 335, 420, 440, 445, 460, 465, 275, 340, 215],
            # Calatayud
            [315, 255, 290, 170, 400, 385, 160, 500, 535, 88, 0, 420, 475, 510, 515, 530, 535, 310, 375, 300],
            # Zaragoza
            [250, 255, 135, 280, 340, 220, 380, 220, 255, 335, 420, 0, 110, 170, 155, 150, 145, 190, 195, 165],
            # Albacete
            [355, 370, 235, 285, 445, 330, 485, 315, 350, 420, 475, 110, 0, 60, 50, 65, 80, 125, 105, 270],  # Villena
            [420, 435, 300, 305, 510, 395, 550, 380, 415, 440, 510, 170, 60, 0, 23, 55, 80, 165, 115, 335],  # Alicante
            [415, 430, 295, 310, 505, 385, 545, 365, 400, 445, 515, 155, 50, 23, 0, 32, 55, 170, 125, 330],  # Elche
            [425, 440, 290, 325, 515, 385, 555, 355, 390, 460, 530, 150, 65, 55, 32, 0, 22, 200, 155, 340],  # Orihuela
            [400, 420, 280, 330, 490, 360, 535, 335, 370, 465, 535, 145, 80, 80, 55, 22, 0, 220, 175, 315],  # Murcia
            [355, 360, 200, 140, 445, 335, 360, 350, 385, 275, 310, 190, 125, 165, 170, 200, 220, 0, 70, 270],
            # Valencia
            [410, 415, 230, 205, 500, 380, 425, 370, 405, 340, 375, 195, 105, 115, 125, 155, 175, 70, 0, 325],  # Gandia
            [85, 105, 82, 230, 175, 100, 250, 170, 205, 215, 300, 165, 270, 335, 330, 340, 315, 270, 325,
             0]]  # Tarancón
        return distancias_carretera[idx_orig][idx_dest]


    def compute_flying_distance(self, ciudad1, ciudad2):
        lat1 = self.gps_positions[ciudad1]["lat"]
        lon1 = self.gps_positions[ciudad1]["lon"]
        lat2 = self.gps_positions[ciudad2]["lat"]
        lon2 = self.gps_positions[ciudad2]["lon"]
        # Radio de la Tierra en kilómetros
        R = 6371.0
        # Convertir grados de latitud y longitud a radianes
        rad_lat1 = math.radians(lat1)
        rad_lon1 = math.radians(lon1)
        rad_lat2 = math.radians(lat2)
        rad_lon2 = math.radians(lon2)
        # Calcular las diferencias entre las coordenadas
        dlat = rad_lat2 - rad_lat1
        dlon = rad_lon2 - rad_lon1
        #  Aplicar la fórmula matemática de Haversine
        a = math.sin(dlat / 2) ** 2 + math.cos(rad_lat1) * math.cos(rad_lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        # Calcular la distancia final
        distancia = R * c
        return distancia


    def plot_network(self, ruta_resaltada=None):
        plt.figure(figsize=(11, 9))

        # 1. Dibujar TODAS las carreteras del grafo en gris
        carreteras_dibujadas = set()
        for node in self.graph.keys():
            for neighbor in self.get_neighbors(node):
                # Evitar dibujar la misma línea dos veces (ida y vuelta)
                identificador_via = tuple(sorted([node, neighbor]))
                if identificador_via not in carreteras_dibujadas:
                    plt.plot([self.gps_positions[node]['lon'],
                              self.gps_positions[neighbor]['lon']],
                             [self.gps_positions[node]['lat'],
                              self.gps_positions[neighbor]['lat']],
                             color='gray', linestyle='-', alpha=0.5, linewidth=1.5, zorder=1)
                    carreteras_dibujadas.add(identificador_via)
                    # Annotate road distance in connected cities
                    road_distance = self.graph[node][neighbor]
                    mid_gps = [0.5 * self.gps_positions[node]['lon'] +
                               0.5 * self.gps_positions[neighbor]['lon'],
                               0.5 * self.gps_positions[node]['lat'] +
                               0.5 * self.gps_positions[neighbor]['lat']]
                    plt.annotate(str(road_distance), (mid_gps[0], mid_gps[1]),
                                 textcoords="offset points",
                                 xytext=(8, 3), fontsize=9, weight='bold')
        # 3. Dibujar los puntos de las ciudades y sus etiquetas
        for node in self.graph.keys():
            # Si la ciudad forma parte de la ruta elegida, se pinta de azul; si no, de rojo
            # color_nodo = 'dodgerblue' if (ruta_resaltada and node.name in ruta_resaltada) else 'red'
            color_nodo = 'red'
            plt.scatter(self.gps_positions[node]['lon'], self.gps_positions[node]['lat'], color=color_nodo,
                        edgecolors='black', s=90, zorder=3)
            plt.annotate(node, (self.gps_positions[node]['lon'], self.gps_positions[node]['lat']),
                         textcoords="offset points",
                         xytext=(8, 3), fontsize=9, weight='bold')

        # Configuración del mapa estático
        plt.title("Red de Carreteras y Ruta Óptima", fontsize=14, pad=15)
        plt.xlabel("Longitud (X, grados)")
        plt.ylabel("Latitud (Y, grados)")
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.gca().set_aspect('equal', adjustable='box')
        if ruta_resaltada:
            plt.legend(loc="upper left")
        plt.show()


class BFS_search():
    def __init__(self, graph):
        self.graph = graph
        self.queue = []
        self.visited_nodes = []
        self.parent_map = {}

    def process_neighbors(self, current_node):
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
                # print('Adding new node to the queue:', neighbor)
                self.visited_nodes.append(neighbor)
                # TODO: QUITAR parent_map --> debe guardar la ruta el alumno
                self.parent_map[neighbor] = current_node
                self.queue.append(neighbor)

    def reorder_queue(self, destination):
        """
        TODO: quitar función, debe añadirla el estudiante.
        Al añadir esta función, ya no es BFS, sino que en la cola se incluye el nodo con mayor interés
        (menor distancia). Al reordenarse, se procesará en la siguiente iteración este nodo.
        :param destination:
        :return:
        """
        distances = []
        for node in self.queue:
            d = self.graph.compute_flying_distance(destination, node)
            distances.append(d)
        sort_index = np.argsort(distances)
        # reorder queue according to current flying distances
        self.queue = [self.queue[i] for i in sort_index]

    def get_route(self, current_node):
        route = []
        total_distance = 0
        while current_node is not None:
            route.append(current_node)
            current_node = self.parent_map[current_node]
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

    def find_route_BFS(self, start_name, destination_name):
        """Finds a route using iterative Breadth-First Search."""
        # Safety check: ensure both cities actually exist in our graph
        if not self.graph.get_node(start_name) or not self.graph.get_node(destination_name):
            return None
        # Standard BFS setup
        self.queue = [start_name]
        self.visited_nodes = [start_name]
        self.parent_map[start_name] = None

        iterations = 0
        while len(self.queue) > 0:
            print(30 * '=')
            print('Current queue is: ', self.queue)
            # pop current_node from the queue
            current_node = self.queue.pop(0)
            print('Current node is: ', current_node)
            if current_node == destination_name:
                print('Found destination! in iterations: ', iterations)
                return self.get_route(current_node)
            self.process_neighbors(current_node)
            iterations += 1
        return None  # No route exists

    def find_route_Heuristic(self, start_name, destination_name):
        """Finds a route using iterative Breadth-First Search."""
        # Safety check: ensure both cities actually exist in our graph
        if not self.graph.get_node(start_name) or not self.graph.get_node(destination_name):
            return None
        # Standard BFS setup
        self.queue = [start_name]
        self.visited_nodes = [start_name]
        self.parent_map[start_name] = None

        iterations = 0
        while len(self.queue) > 0:
            print(30 * '=')
            print('Current queue is: ', self.queue)
            # pop current_node from the queue
            current_node = self.queue.pop(0)
            print('Current node is: ', current_node)
            if current_node == destination_name:
                print('Found destination! in iterations: ', iterations)
                # Found solution: destination reached --> reconstruct the route
                return self.get_route(current_node)
            self.process_neighbors(current_node)
            # TODO: CUIDADO! DEBEMOS REORDENAR LA LISTA DE ACUERDO CON NUESTRA HEURÍSTICA
            self.reorder_queue(destination_name)
            iterations += 1
        return None  # No route exists


if __name__ == "__main__":
    # --- 1. Initialize the Graph ---
    spain_network = Graph()
    spain_network.build_network()
    print(spain_network.graph)
    spain_network.plot_network()

    # Para que todo quede más limpio, tenemos una clase para la búsqueda
    algoritmo = BFS_search(spain_network)
    # --- 3. Run the Search ---
    origen = 'Madrid'
    destino = 'Murcia'
    route, distance = algoritmo.find_route_Heuristic(origen , destino)

    # --- 4. Print Results ---
    if route:
        print(f"Route found: {' -> '.join(route)}")
        print(f"TOTAL DISTANCE IS: ", distance)
    else:
        print(f"No route found between start and end.")

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
