from graph.graph import Graph
from search_algorithms.bfs_algorithm import BFS_Algorithm
from search_algorithms.greedy_algorithm import Greedy_Algorithm
from search_algorithms.dijskstra_algorithm import Dijkstra_Algorithm
from search_algorithms.a_star_algorithm import A_star_Algorithm


if __name__ == "__main__":
    spain_network = Graph()
    spain_network.build_network()
    # spain_network.build_network2()
    ##################################################################
    # EJERCICIO 2.1: OBSERVE EL GRAFO CON UN PRINT O EL DEBUGGER
    ##################################################################
    print(spain_network.graph)
    spain_network.plot_network()

    ##################################################################
    # EJERCICIO 2.2: EJECUTE get_node(self, name), get_distance, ... etc
    # sobre el grafo.
    ##################################################################
    # print(spain_network.get_node('Madrid'))
    # print(spain_network.get_distance('Madrid', 'Cuenca'))
    # print(spain_network.get_neighbors('Elx'))
    # print(spain_network.compute_flying_distance('Madrid', 'Elx'))

    ##################################################################
    # EJERCICIO 2.7: cambie el origen y el destino y anote toda
    # la información relevante.
    ##################################################################
    # algoritmo = BFS_Algorithm(spain_network)
    # origen = 'Madrid'
    # destino = 'Elx'
    # route, distance, iterations = algoritmo.find_route(origen, destino)

    ##################################################################
    # EJERCICIO 2.9: pruebe a buscar una ruta entre Elche-Madrid y
    # Madrid-Elche. ¿Es óptima, es igual?
    ##################################################################
    # algoritmo = Greedy_Algorithm(spain_network)
    # origen = 'Madrid'
    # destino = 'Elx'
    # route, distance, iterations = algoritmo.find_route(origen , destino)

    ##################################################################
    # EJERCICIO 2.13: PRUEBE DIFERENTES ORÍGENES Y DESTINOS
    ##################################################################
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
        print('In ', iterations, ' iterations')
        print(f"THE TOTAL DISTANCE IS: ", distance)
    else:
        print(f"No route found between start and end.")


