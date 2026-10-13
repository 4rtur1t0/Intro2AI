from graph.graph import Graph
from search_algorithms.bfs_algorithm import BFS_Algorithm
from search_algorithms.greedy_algorithm import Greedy_Algorithm
from search_algorithms.dijskstra_algorithm import Dijkstra_Algorithm
from search_algorithms.a_star_algorithm import A_star_Algorithm


if __name__ == "__main__":
    spain_network = Graph()
    spain_network.build_network()
    # spain_network.build_network2()
    print(spain_network.graph)
    spain_network.plot_network()

    # EXERCICI 2.2
    # print(spain_network.get_node('Madrid'))
    # print(spain_network.get_distance('Madrid', 'Cuenca'))
    # print(spain_network.get_neighbors('Elx'))
    # print(spain_network.compute_flying_distance('Madrid', 'Elx'))

    # algoritmo = BFS_Algorithm(spain_network)
    # origen = 'Madrid'
    # destino = 'Elx'
    # route, distance, iterations = algoritmo.find_route(origen, destino)

    # algoritmo = Greedy_Algorithm(spain_network)
    # origen = 'Madrid'
    # destino = 'Elx'
    # route, distance, iterations = algoritmo.find_route(origen , destino)

    algoritmo = A_star_Algorithm(spain_network)
    origen = 'Madrid'
    destino = 'Elx'
    route, distance, iterations = algoritmo.find_route(origen, destino)

   # algoritmo = Dijkstra_Algorithm(spain_network)
   # origen = 'Madrid'
   # destino = 'Villena'
   # route, distance, iterations = algoritmo.find_route(origen, destino)

    # Print Results ---
    if route:
        print(f"Route found: {' -> '.join(route)}")
        print(f"TOTAL DISTANCE IS: ", distance)
    else:
        print(f"No route found between start and end.")


