from graph.graph import Graph
from search_algorithms.bfs_algorithm import BFS_Algorithm
from search_algorithms.greedy_algorithm import Greedy_Algorithm
from search_algorithms.a_star_algorithm import A_star_Algorithm

if __name__ == "__main__":
    spain_network = Graph()
    spain_network.build_network()
    print(spain_network.graph)
    spain_network.plot_network()

    algoritmo = BFS_Algorithm(spain_network)
    origen = 'Madrid'
    destino = 'Elche'
    route, distance, iterations = algoritmo.find_route(origen, destino)

    # algoritmo = Greedy_Algorithm(spain_network)
    # origen = 'Madrid'
    # destino = 'Elche'
    # route, distance, iterations = algoritmo.find_route(origen , destino)

    # algoritmo = A_star_Algorithm(spain_network)
    # origen = 'Madrid'
    # destino = 'Elche'
    # route, distance, iterations = algoritmo.find_route(origen, destino)

    # Print Results ---
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
