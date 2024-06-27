import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import json
from tabulate import tabulate


def generate_cities(num_cities, seed=42):
    np.random.seed(seed)
    cities = np.random.rand(num_cities, 2) * 100
    return cities


def generate_connected_graph(nodes, edgelimit=5):
    num_nodes = len(nodes)
    G = nx.Graph()
    distances = cdist(nodes, nodes, 'euclidean')
    mst = nx.minimum_spanning_tree(nx.Graph(distances))

    for u, v, data in mst.edges(data=True):
        G.add_edge(u, v, weight=round(data['weight'], 1))

    potential_edges = [(u, v) for u in range(num_nodes) for v in range(u + 1, num_nodes) if not G.has_edge(u, v)]
    np.random.shuffle(potential_edges)

    added_edges = 0
    for u, v in potential_edges:
        if G.degree[u] < edgelimit and G.degree[v] < edgelimit:
            G.add_edge(u, v, weight=round(distances[u, v], 1))
            added_edges += 1
            if added_edges >= edgelimit * num_nodes // 2:
                break

    return G


def calculate_shortest_paths(G):
    return dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))


def build_complete_graph(shortest_paths, num_nodes):
    complete_graph = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            complete_graph[i, j] = shortest_paths[i][j]
    return complete_graph


def generate_capacities(num_cities, min_demand=1, max_demand=10, depot_index=0):
    capacities = np.random.randint(min_demand, max_demand + 1, size=num_cities)
    capacities[depot_index] = 0
    return capacities.tolist()


def save_to_file(cities, distances, capacities, filename):
    # data = {
    #     "cities": cities.tolist(),
    #     "distances": distances.tolist(),
    #     "demandes": capacities
    # }
    data = f"{len(cities).tolist()}\n{distances.tolist()}\n{capacities}"
    # with open(filename, 'w') as f:
    #     json.dump(data, f)
    with open(filename, 'w') as f:
        f.write(data)


def plot_graph(G, cities):
    pos = {i: (city[0], city[1]) for i, city in enumerate(cities)}
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_size=700, node_color="skyblue", font_size=10, font_weight="bold")
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=8)
    plt.title("Graph des villes avec les distances pondérées")
    plt.show()

def demande_value():
    while True:
        while True:
            min_num_cities = input("Le nombre minimum de cities : ")
            try:
                min_num_cities = int(min_num_cities)
            except:
                print("Le nombre minimum de cities n'est pas valide ")
                #min_num_cities = input("Le nombre minimum de cities : ")
            else:
                print(f"Le nombre minimum de cities {min_num_cities}")

                while True:
                    max_num_cities = input("Le nombre maximum de cities : ")
                    try:
                        max_num_cities = int(max_num_cities)
                    except:
                        print("Le nombre maximum de cities n'est pas valide ")
                        #max_num_cities = input("Le nombre maximum de cities : ")
                    else:
                        print(f"Le nombre maximum de cities {max_num_cities}")
                        while True:
                            min_avg_degree = input("Le degree moyen minimum de cities : ")
                            try:
                                min_avg_degree = int(min_avg_degree)
                            except:
                                print("Le nombre")
                                #min_avg_degree = input("Le degree moyen minimum de cities : ")
                            else:
                                print(f"Le degree moyen minimum de cities : {min_avg_degree}")
                                while True:
                                    max_avg_degree = input("le degree moyen maximum de cities : ")
                                    try:
                                        max_avg_degree = int(max_avg_degree)
                                    except:
                                        print("Le nombre")
                                        #max_avg_degree = input("le nombre")
                                    else:
                                        print(f"Le degree moyen maximum {max_avg_degree}")
                                        while True:
                                            limit_edge = input("Le nombre limit d'arret par sommet : ")
                                            try:
                                                limit_edge = int(limit_edge)
                                            except:
                                                print("Le nombre limit d'arret par sommet n'est pas valide ")
                                                #limit_edge = input("Le nombre limit d'arret par sommet : ")
                                            else:
                                                print(f"Le nombre limit de arret {limit_edge}")
                                                while True:
                                                    min_demande_client = input("Le nombre minimum de demande des clients : ")
                                                    try:
                                                        min_demande_client = int(min_demande_client)
                                                    except:
                                                        print("Le nombre minimum de demande des clients n'est pas valide ")
                                                        #min_demande_client = input("Le nombre minimum de demande des clients : ")
                                                    else:
                                                        print(f"Le nombre minimum de demande {min_demande_client}")
                                                        while True:
                                                            max_demande_client = input("Le nombre maximal de demande des clients : ")
                                                            try:
                                                                max_demande_client = int(max_demande_client)
                                                            except:
                                                                print("Le nombre maximal de demande des clients est invalide")
                                                                #max_demande_client = input("Le nombre maximal de demande des clients : ")
                                                            else:
                                                                print(f"Le nombre maximal de demande {max_demande_client}")

                                                            break

        break
    return min_num_cities, max_num_cities,min_avg_degree, max_avg_degree, limit_edge, min_demande_client, max_demande_client



num_cities = np.random.randint(50, 100)
avg_degree = np.random.uniform(1, 4)
limit = 5
min_demand = 4
max_demand = 30
depot_index = 0

cities = generate_cities(num_cities)
G = generate_connected_graph(cities, limit)
shortest_paths = calculate_shortest_paths(G)
complete_graph = build_complete_graph(shortest_paths, num_cities)
capacities = generate_capacities(num_cities, min_demand, max_demand, depot_index)

save_to_file(cities, complete_graph, capacities, f'graphes/{num_cities}nodes_{depot_index}dindex.json')

print("Coordonnées des villes :\n", cities)
print("Capacités des villes :\n", capacities)
print("Graphe des villes :\n", G)
print("Nombre de villes : ", num_cities)
print("Degré moyen : ", avg_degree)
print("Données sauvegardées dans 'vrp_data.json'")

table = [
    ["Nombre de villes", num_cities],
    ["Degré moyen", avg_degree],
    ["Limite d'arêtes par sommet", limit],
    ["Capacité minimale", min_demand],
    ["Capacité maximale", max_demand],
    ["Index du dépôt", depot_index]
]
print("\nParamètres de génération du VRP :")
print(tabulate(table, headers=["Paramètre", "Valeur"], tablefmt="pretty"))

plot_graph(G, cities)
