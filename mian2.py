import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import json
from tabulate import tabulate

# Générer des coordonnées aléatoires pour les villes
# num_cities : Nombre de villes (sommets)
# seed : Graine pour le générateur de nombres aléatoires pour reproduire les résultats
def generate_cities(num_cities, seed=42):
    np.random.seed(seed)
    cities = np.random.rand(num_cities, 2) * 100
    return cities


# Générer un graphe connexe avec des distances réalistes entre les villes
# cities : Coordonnées des villes
# limit : Limite du nombre d'arêtes par sommet
# min_weight, max_weight : Bornes pour les poids des arêtes
# cities : Coordonnées des villes
def generate_connected_graph(nodes, edgelimit=5):
    num_nodes = len(nodes)
    G = nx.Graph()

    # Calculer les distances euclidiennes entre les villes
    distances = cdist(nodes, nodes, 'euclidean')

    # Ajouter les arêtes pour former un arbre couvrant minimal (MST) pour garantir que le graphe est connexe
    mst = nx.minimum_spanning_tree(nx.Graph(distances))

    for u, v, data in mst.edges(data=True):
        G.add_edge(u, v, weight=round(data['weight'], 1))

    # Ajouter des arêtes supplémentaires aléatoires pour atteindre le degré moyen souhaité tout en évitant les doublons
    while nx.is_connected(G) and G.number_of_edges() < num_nodes * (num_nodes - 1) // 2:
        u, v = np.random.choice(num_nodes, 2, replace=False)
        if not G.has_edge(u, v) and G.degree[u] < edgelimit and G.degree[v] < edgelimit:
            G.add_edge(u, v, weight=round(distances[u, v], 1))

    return G

# Calculer les plus courts chemins dans le graphe d'origine
# G : Le graphe des villes
def calculate_shortest_paths(G):
    return dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))


# Construire le graphe complet avec les longueurs des plus courts chemins
# shortest_paths : Dictionnaire des plus courts chemins
# num_nodes : Nombre de villes (sommets)
def build_complete_graph(shortest_paths, num_nodes):
    complete_graph = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            complete_graph[i, j] = shortest_paths[i][j]
    return complete_graph


# Générer les capacités pour les villes
# num_cities : Nombre de villes
# min_demand, max_demand : Bornes pour les capacités
# depot_index : Index du dépôt (capacité 0)
def generate_capacities(num_cities, min_demand=1, max_demand=10, depot_index=0):
    capacities = np.random.randint(min_demand, max_demand + 1, size=num_cities)
    capacities[depot_index] = 0
    return capacities.tolist()


# Sauvegarder les données des villes et des distances dans un fichier JSON
# cities : Coordonnées des villes
# distances : Matrice des distances
# filename : Nom du fichier de sortie
def save_to_file(cities, distances, capacities, filename):
    data = {
        "cities": cities.tolist(),
        "distances": distances.tolist(),
        "capacities": capacities
    }
    with open(filename, 'w') as f:
        json.dump(data, f)


# Visualiser le graphe
# G : Le graphe des villes
# cities : Coordonnées des villes
def plot_graph(G, cities):
    pos = {i: (city[0], city[1]) for i, city in enumerate(cities)}
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_size=700, node_color="skyblue", font_size=10, font_weight="bold")
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=8)
    plt.title("Graph des villes avec les distances pondérées")
    plt.show()


# Exemple d'utilisation
num_cities = np.random.randint(5, 20)  # Nombre de sommets (villes) entre 5 et 14
avg_degree = np.random.uniform(1, 4)  # Degré moyen aléatoire réaliste

# Paramètres supplémentaires
limit = 5  # Limite du nombre d'arêtes par sommet
min_weight = 1  # Poids minimum pour les arêtes
max_weight = 30  # Poids maximum pour les arêtes
min_demand = 4  # Capacité minimale pour les villes
max_demand = 30  # Capacité maximale pour les villes
depot_index = 0  # Index du dépôt

# Génération des villes et du graphe connexe
cities = generate_cities(num_cities)
G = generate_connected_graph(cities, limit)
shortest_paths = calculate_shortest_paths(G)
complete_graph = build_complete_graph(shortest_paths, num_cities)
capacities = generate_capacities(num_cities, min_demand, max_demand, depot_index)

# Sauvegarder les données dans un fichier JSON
save_to_file(cities, complete_graph, capacities, 'vrp_data.json')

# Afficher quelques informations pour validation
print("Coordonnées des villes :\n", cities)
print("Capacités des villes :\n", capacities)
print("Graphe des villes :\n", G)
print("Nombre de villes : ", num_cities)
print("Degré moyen : ", avg_degree)
print("Données sauvegardées dans 'vrp_data.json'")

# Imprimer les informations sous forme de tableau
table = [
    ["Nombre de villes", num_cities],
    ["Degré moyen", avg_degree],
    ["Limite d'arêtes par sommet", limit],
    ["Poids minimum des arêtes", min_weight],
    ["Poids maximum des arêtes", max_weight],
    ["Capacité minimale", min_demand],
    ["Capacité maximale", max_demand],
    ["Index du dépôt", depot_index]
]
print("\nParamètres de génération du VRP :")
print(tabulate(table, headers=["Paramètre", "Valeur"], tablefmt="pretty"))
# print(table)
# Visualiser le graphe
plot_graph(G, cities)