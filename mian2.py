import numpy as np
from random import randint
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


def generate_truck_capacities(num_trucks, min_capacity=40, max_capacity=100):
    return np.random.randint(min_capacity, max_capacity + 1, size=num_trucks).tolist()


def save_to_file(cities, distances, capacities, filename, truck_capacities):
    data = f"{len(cities)}\n{distances.tolist()}\n{capacities}\n{truck_capacities}"
    with open(filename, 'w') as f:
        f.write(data)


def save_to_file_json(cities, distances, capacities, filename, truck_capacities):
    data = {
        "cities": cities.tolist(),
        "distances": distances.tolist(),
        "demandes": capacities,
        "trucks_capacities": truck_capacities
    }
    with open(filename, 'w') as f:
        json.dump(data, f)


def plot_graph(G, cities):
    pos = {i: (city[0], city[1]) for i, city in enumerate(cities)}
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_size=700, node_color="skyblue", font_size=10, font_weight="bold")
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=8)
    plt.title("Graph des villes avec les distances pondérées")
    plt.show()


num_cities = np.random.randint(10, 25)
avg_degree = np.random.uniform(1, 4)
limit = 5
min_demand = 4
max_demand = 30
depot_index = 0
num_trucks = randint(1, 5)
min_truck_capacity = 40
max_truck_capacity = 100

cities = generate_cities(num_cities)
G = generate_connected_graph(cities, limit)
shortest_paths = calculate_shortest_paths(G)
complete_graph = build_complete_graph(shortest_paths, num_cities)
capacities = generate_capacities(num_cities, min_demand, max_demand, depot_index)
truck_capacities = generate_truck_capacities(num_trucks, min_truck_capacity, max_truck_capacity)

save_to_file(cities, complete_graph, capacities,
             f'graphes/{num_cities}nodes_{depot_index}dindex_{num_trucks}trucks.txt', truck_capacities)
save_to_file_json(cities, complete_graph, capacities,
                  f'graphes/{num_cities}nodes_{depot_index}dindex_{num_trucks}trucks.json', truck_capacities)

print("Coordonnées des villes :\n", cities)
print("Capacités des villes :\n", capacities)
print("Capacités des camions :\n", truck_capacities)
print("Graphe des villes :\n", G)
print("Nombre de villes : ", num_cities)
print("Degré moyen : ", avg_degree)

table = [
    ["Nombre de villes", num_cities],
    ["Degré moyen", avg_degree],
    ["Limite d'arêtes par sommet", limit],
    ["Demande minimale", min_demand],
    ["Demande maximale", max_demand],
    ["Capacités des camions", truck_capacities],
    ["Nombre de véhicules", num_trucks],
    ["Index du dépôt", depot_index]
]
print("\nParamètres de génération du VRP :")
print(tabulate(table, headers=["Paramètre", "Valeur"], tablefmt="pretty"))

plot_graph(G, cities)
