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
        self.add_edge("Madrid", "Ciudad Real")
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
        "Gandia": {"lat": 38.9680, "lon": -0.1830}}

    def road_distance_data(self, origen, destino):
        # Ciudades en orden:
        # 0: Madrid, 1: Guadalajara, 2: Cuenca, 3: Teruel, 4: Segovia, 5: Toledo, 6: Soria,
        # 7: Ciudad Real, 8: Puertollano, 9: Calatayud, 10: Zaragoza, 11: Albacete,
        # 12: Villena, 13: Alicante, 14: Elche, 15: Orihuela, 16: Murcia, 17: Valencia, 18: Gandia
        ciudades = [
            "Madrid", "Guadalajara", "Cuenca", "Teruel", "Segovia", "Toledo", "Soria",
            "Ciudad Real", "Puertollano", "Calatayud", "Zaragoza", "Albacete",
            "Villena", "Alicante", "Elche", "Orihuela", "Murcia", "Valencia", "Gandia" ]
        idx_orig = ciudades.index(origen)
        idx_dest = ciudades.index(destino)
        distancias_carretera = [
            # Mad  Gua  Cue  Ter  Seg  Tol  Sor  CR   Pue  Cal  Zar  Alb  Vil  Ali  Elc  Ori  Mur  Val  Gan
            [0, 58, 167, 302, 87, 71, 231, 190, 240, 235, 320, 258, 360, 420, 415, 402, 401, 352, 415],  # Madrid
            [58, 0, 135, 248, 145, 130, 175, 250, 300, 180, 265, 255, 365, 425, 420, 415, 410, 335, 405],  # Guadalajara
            [167, 135, 0, 150, 250, 180, 245, 245, 290, 170, 260, 145, 250, 310, 305, 312, 310, 200, 260],  # Cuenca
            [302, 248, 150, 0, 385, 345, 225, 370, 415, 135, 175, 245, 295, 320, 325, 330, 345, 140, 200],  # Teruel
            [87, 145, 250, 385, 0, 155, 190, 275, 325, 320, 405, 345, 445, 505, 500, 490, 485, 440, 500],  # Segovia
            [71, 130, 180, 345, 155, 0, 300, 120, 170, 305, 390, 245, 345, 405, 400, 375, 365, 370, 410],  # Toledo
            [231, 175, 245, 225, 190, 300, 0, 420, 470, 90, 160, 390, 495, 545, 540, 545, 550, 360, 425],  # Soria
            [190, 250, 245, 370, 275, 120, 420, 0, 45, 375, 460, 220, 320, 355, 350, 315, 305, 345, 365],  # Ciudad Real
            [240, 300, 290, 415, 325, 170, 470, 45, 0, 420, 505, 245, 345, 380, 375, 335, 325, 370, 390],  # Puertollano
            [235, 180, 170, 135, 320, 305, 90, 375, 420, 0, 85, 310, 410, 435, 440, 445, 455, 275, 335],  # Calatayud
            [320, 265, 260, 175, 405, 390, 160, 460, 505, 85, 0, 395, 495, 520, 525, 530, 540, 315, 375],  # Zaragoza
            [258, 255, 145, 245, 345, 245, 390, 220, 245, 310, 395, 0, 105, 165, 160, 150, 145, 190, 195],  # Albacete
            [360, 365, 250, 295, 445, 345, 495, 320, 345, 410, 495, 105, 0, 60, 40, 48, 65, 130, 115],  # Villena
            [420, 425, 310, 320, 505, 405, 545, 355, 380, 435, 520, 165, 60, 0, 25, 55, 75, 165, 110],  # Alicante
            [415, 420, 305, 325, 500, 400, 540, 350, 375, 440, 525, 160, 40, 25, 0, 33, 55, 170, 125],  # Elche
            [402, 415, 312, 330, 490, 375, 545, 315, 335, 445, 530, 150, 48, 55, 33, 0, 22, 195, 155],  # Orihuela
            [401, 410, 310, 345, 485, 365, 550, 305, 325, 455, 540, 145, 65, 75, 55, 22, 0, 215, 180],  # Murcia
            [352, 335, 200, 140, 440, 370, 360, 345, 370, 275, 315, 190, 130, 165, 170, 195, 215, 0, 70],  # Valencia
            [415, 405, 260, 200, 500, 410, 425, 365, 390, 335, 375, 195, 115, 110, 125, 155, 180, 70, 0]  # Gandia
        ]
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
        a = math.sin(dlat / 2)**2 + math.cos(rad_lat1) * math.cos(rad_lat2) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        # Calcular la distancia final
        distancia = R * c
        return distancia

    def plot_network(self):
        # 2. Separar los datos en listas para Matplotlib
        nombres = list(self.gps_positions.keys())
        longitudes = [self.gps_positions[c]["lon"] for c in nombres]
        latitudes = [self.gps_positions[c]["lat"] for c in nombres]

        # 3. Configurar el lienzo de la gráfica
        plt.figure(figsize=(10, 8))

        # 4. Dibujar las ciudades como puntos (X = Longitud, Y = Latitud)
        plt.scatter(longitudes, latitudes, color='red', edgecolors='black', s=80, zorder=5)

        # 5. Añadir los nombres al lado de cada punto
        for i, nombre in enumerate(nombres):
            plt.annotate(
                nombre,
                (longitudes[i], latitudes[i]),
                textcoords="offset points",
                xytext=(8, 3),  # Separa el texto un poco a la derecha del punto
                fontsize=9,
                weight='bold'
            )

        # 6. Personalizar los ejes y cuadrícula
        plt.title("Mapa Aproximado de Ciudades (Proyección GPS Plana)", fontsize=14, pad=15)
        plt.xlabel("Longitud (Eje X)", fontsize=11)
        plt.ylabel("Latitud (Eje Y)", fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.5)

        # Forzar una proporción de aspecto idéntica para evitar que el mapa se estire
        plt.gca().set_aspect('equal', adjustable='box')

        # 7. Mostrar la gráfica en pantalla
        plt.show()

    def graficar_red(self, ruta_resaltada=None):
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

        # # 2. Si hay una ruta calculada, resaltarla en azul grueso
        # if ruta_resaltada and len(ruta_resaltada) > 1:
        #     for i in range(len(ruta_resaltada) - 1):
        #         nodo_a = self.all_nodes[ruta_resaltada[i]]
        #         nodo_b = self.all_nodes[ruta_resaltada[i + 1]]
        #         plt.plot([nodo_a.lon, nodo_b.lon], [nodo_a.lat, nodo_b.lat],
        #                  color='blue', linestyle='-', linewidth=4, alpha=0.8, zorder=2,
        #                  label="Ruta Calculada" if i == 0 else "")

        # 3. Dibujar los puntos de las ciudades y sus etiquetas
        for node in self.graph.keys():
            # Si la ciudad forma parte de la ruta elegida, se pinta de azul; si no, de rojo
            #color_nodo = 'dodgerblue' if (ruta_resaltada and node.name in ruta_resaltada) else 'red'
            color_nodo = 'red'
            plt.scatter(self.gps_positions[node]['lon'], self.gps_positions[node]['lat'], color=color_nodo, edgecolors='black', s=90, zorder=3)
            plt.annotate(node, (self.gps_positions[node]['lon'], self.gps_positions[node]['lat']), textcoords="offset points",
                         xytext=(8, 3), fontsize=9, weight='bold')

        # Configuración del mapa estático
        plt.title("Red de Carreteras y Ruta Óptima", fontsize=14, pad=15)
        plt.xlabel("Longitud (X)")
        plt.ylabel("Latitud (Y)")
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

    def process_neighbors_bfs1(self, current_node):
        # Get the list of neighbors of the current node
        neighbors = self.graph.get_neighbors(current_node)
        print('Found neighbors:', neighbors)
        # para cada vecino neighbour encontrado
        for neighbor in neighbors:
            if neighbor not in self.visited_nodes:
                # si no se ha visitado: a) se añade a la lista de visitados y b) se añade a la cola de exploración
                #print('Adding new node to the queue:', neighbor)
                self.visited_nodes.append(neighbor)
                self.parent_map[neighbor] = current_node
                self.queue.append(neighbor)

    # def process_neighbors_bfs2(self, current_node):
    #     # Get the list of neighbors of the current node
    #     neighbors = self.graph.get_neighbors(current_node)
    #     print('Found neighbors:', neighbors)
    #     # DEBE COMPLETARSE POR EL ALUMNO
    #     # SE DEBE GUARDAR LA RUTA
    #     return neighbors

    def reorder_queue(self, destination):
        distances = []
        for node in self.queue:
            d = self.graph.compute_flying_distance(destination, node)
            distances.append(d)
        sort_index = np.argsort(distances)
        # reorder queue according to current flying distances
        self.queue = [self.queue[i] for i in sort_index]


    def get_route(self, current_node):
        route = []
        while current_node is not None:
            route.append(current_node)
            current_node = self.parent_map[current_node]
        return route[::-1]  # Reverse to get start -> destination

    def find_route_BFS1(self, start_name, destination_name):
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
            print(30*'=')
            print('Current queue is: ', self.queue)
            # pop current_node from the queue
            current_node = self.queue.pop(0)
            print('Current node is: ', current_node)
            if current_node == destination_name:
                print('Found destination! in iterations: ', iterations)
                return self.get_route(current_node)
            self.process_neighbors_bfs1(current_node)
            iterations += 1
        return None  # No route exists

    # def find_route_BFS2(self, start_name, destination_name):
    #     """Finds a route using iterative Breadth-First Search."""
    #     # Safety check: ensure both cities actually exist in our graph
    #     if not self.graph.get_node(start_name) or not self.graph.get_node(destination_name):
    #         return None
    #     # Standard BFS setup
    #     self.queue = [start_name]
    #     self.visited_nodes = [start_name]
    #     self.parent_map[start_name] = None
    #
    #     while len(self.queue) > 0:
    #         print(30*'=')
    #         print('Current queue is: ', self.queue)
    #         # pop current_node from the queue
    #         current_node = self.queue.pop(0)
    #         print('Current node is: ', current_node)
    #         if current_node == destination_name:
    #             # Found solution: destination reached --> reconstruct the route
    #             return self.get_route(current_node)
    #         # CUIDADO! SE DEBE COMPLETAR ESTA FUNCIÓN
    #         self.process_neighbors_bfs2(current_node)
    #     return None  # No route exists

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
            print(30*'=')
            print('Current queue is: ', self.queue)
            # pop current_node from the queue
            current_node = self.queue.pop(0)
            print('Current node is: ', current_node)
            if current_node == destination_name:
                print('Found destination! in iterations: ', iterations)
                # Found solution: destination reached --> reconstruct the route
                return self.get_route(current_node)
            self.process_neighbors_bfs1(current_node)
            # CUIDADO! DEBEMOS REORDENAR LA LISTA DE ACUERDO CON NUESTRA HEURÍSTICA
            self.reorder_queue(destination_name)
            iterations += 1
        return None  # No route exists


if __name__ == "__main__":
    # --- 1. Initialize the Graph ---
    spain_network = Graph()
    spain_network.build_network()
    #spain_network.plot_network()
    spain_network.graficar_red()

    # Para que todo quede más limpio, tenemos una clase para la búsqueda
    algoritmo = BFS_search(spain_network)
    # --- 3. Run the Search ---
    #route = algoritmo.find_route_BFS1('Madrid', 'Elche')
    #route = algoritmo.find_route_BFS2('Madrid', 'Elche')
    route = algoritmo.find_route_Heuristic('Madrid', 'Elche')

    # --- 4. Print Results ---
    if route:
        print(f"Route found: {' -> '.join(route)}")
    else:
        print(f"No route found between start and end.")






    #BFS_search1
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

    #BFS_search2
    # sin bucles infinitos
    # guarda la ruta
    # la ruta no es óptima en términos de distancia


    # Dijkstra

    # A*

    # Comparación



