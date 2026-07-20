import numpy as np

class Graph():
    def __init__(self):
        self.graph = {}
    # Function to add connections (undirected roads)
    def add_edge(self, node1, node2, distance):
        # Ensure city1 exists in the graph
        if node1 not in self.graph:
            self.graph[node1] = {}
        # Ensure city2 exists in the graph
        if node2 not in self.graph:
            self.graph[node2] = {}

        # Connect both cities (two-way road)
        self.graph[node1][node2] = distance
        self.graph[node2][node1] = distance
        #print(self.graph)

    def create_map(self):
        # 3. Build your city network
        self.add_edge("Madrid", "Guadalajara", 66)
        self.add_edge("Madrid", "Cuenca", 193)
        self.add_edge("Madrid", "Segovia", 98)
        self.add_edge("Madrid", "Toledo", 73)
        self.add_edge("Guadalajara", "Calatayud", 175)
        self.add_edge("Cuenca", "Albacete", 164)
        self.add_edge("Albacete", "Villena", 114)
        self.add_edge("Villena", "Alicante", 57)
        self.add_edge("Alicante", "Elche", 25)

    def get_distance(self, node1, node2):
        # 4. Look up a distance directly
        distance = self.graph[node1][node2]
        print(f"Distance from nodes: {distance} km")

    def goal_reached(self, node, goal):
        if node == goal:
            return True
        else:
            return False

    def successors(self, node):
        # Find all connections for a city (node)
        successors = []
        #print("\nConnections from node:")
        for neighbor, distance in self.graph[node].items():
            #print(f"- To {neighbor}: {distance} km")
            successors.append(neighbor)
        return successors

    def BFS_search1(self, nodelist, goal):
        """
        BFS always guarantees the shortest path in terms of the number of stops (hops) between two cities.
        Pero no siempre encuentra la distancia mínima
        """
        new_nodes = []
        for node in nodelist:
            if self.goal_reached(node, goal):
                return True, node
            # extend es similar a append, pero no hace sublistas: prueba append en vez de extend!
            new_nodes.extend(self.successors(node))
        if len(new_nodes) != 0:
            # make it unique
            new_nodes = list(set(new_nodes))
            return self.BFS_search(new_nodes, goal)
        else:
            return False, None

    def BFS_search2(self, nodelist, goal):
        """
        BFS always guarantees the shortest path in terms of the number of stops (hops) between two cities.
        Pero no siempre encuentra la distancia mínima
        """
        new_nodes = []
        for node in nodelist:
            if self.goal_reached(node, goal):
                return True, node
            # extend es similar a append, pero no hace sublistas: prueba append en vez de extend!
            new_nodes.extend(self.successors(node))
        if len(new_nodes) != 0:
            # make it unique
            new_nodes = list(set(new_nodes))
            return self.BFS_search(new_nodes, goal)
        else:
            return False, None


if __name__ == "__main__":
    grafo = Graph()
    grafo.create_map()
    print(grafo.graph)
    result = grafo.BFS_search1(['Madrid'], 'Elche')
    result = grafo.BFS_search2(['Madrid'], 'Elche')
    print(result)



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



