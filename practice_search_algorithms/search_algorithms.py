from graph.graph import Graph
from search_algorithms.bfs_algorithm import BFS_Algorithm
from search_algorithms.greedy_algorithm import Greedy_Algorithm
from search_algorithms.dijskstra_algorithm import Dijkstra_Algorithm
from search_algorithms.a_star_algorithm import A_star_Algorithm


if __name__ == "__main__":
    spain_network = Graph()
    spain_network.build_network()
    # spain_network.build_network2()
    #################################
    # EJERCICIO 2.1: OBSERVE EL GRAFO
    #################################
    print(spain_network.graph)
    spain_network.plot_network()

    #################################
    # EJERCICIO 2.2: EJECUTE get_node(self, name), get_distance, ... etc
    # sobre el grafo.
    #################################

    #########################################################
    # Descomente uno de los algoritmos cada vez
    ########################################################
    #algoritmo = BFS_Algorithm(spain_network)
    #origen = 'Madrid'
    #destino = 'Elx'
    #route, distance, iterations = algoritmo.find_route(origen, destino)

    algoritmo = Greedy_Algorithm(spain_network)
    origen = 'Madrid'
    destino = 'Elx'
    route, distance, iterations = algoritmo.find_route(origen , destino)

    # algoritmo = A_star_Algorithm(spain_network)
    # origen = 'Madrid'
    # destino = 'Elx'
    # route, distance, iterations = algoritmo.find_route(origen, destino)

    # Print Results ---
    if route:
        print(f"Route found: {' -> '.join(route)}")
        print('In ', iterations, ' iterations')
        print(f"THE TOTAL DISTANCE IS: ", distance)
    else:
        print(f"No route found between start and end.")


