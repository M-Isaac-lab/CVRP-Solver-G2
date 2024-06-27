import random

def mutate(solution):
    # Sélectionner une route aléatoire pour la mutation
    route_index = random.randint(0, len(solution.routes) - 1)
    route = solution.routes[route_index]

    # Sélectionner deux clients aléatoires dans la route
    client1_index = random.randint(0, len(route) - 1)
    client2_index = random.randint(0, len(route) - 1)
    while client1_index == client2_index:
        client2_index = random.randint(0, len(route) - 1)

    # Échanger les positions des deux clients sélectionnés
    route[client1_index], route[client2_index] = route[client2_index], route[client1_index]

    # Vérifier la faisabilité de la solution
    if is_solution_feasible(solution, route):
        # Recalculer la distance totale de la solution
        solution.total_distance = calculate_total_distance(solution.routes)
        return solution
    else:
        # Si la solution n'est pas faisable, annuler la mutation
        route[client1_index], route[client2_index] = route[client2_index], route[client1_index]
        return solution

def is_solution_feasible(solution, route):
    current_capacity = 0
    for client in route:
        current_capacity += client.demand
        if current_capacity > solution.capacities[route.index(client)]:
            return False
    return True

def calculate_total_distance(routes):
    total_distance = 0
    for route in routes:
        for i in range(len(route)):
            if i == 0:
                total_distance += route[0].distance_to(route[-1])
            else:
                total_distance += route[i-1].distance_to(route[i])
    return total_distance





class Client:
    def __init__(self, id, demand, distance_to_depot):
        self.id = id
        self.demand = demand
        self.distance_to_depot = distance_to_depot

    def distance_to(self, other_client):
        return abs(self.id - other_client.id)


class Solution:
    def __init__(self, routes, capacities):
        self.routes = routes
        self.capacities = capacities
        self.total_distance = calculate_total_distance(routes)


def calculate_total_distance(routes):
    total_distance = 0
    for route in routes:
        for i in range(len(route)):
            if i == 0:
                total_distance += route[0].distance_to_depot
            else:
                total_distance += route[i - 1].distance_to(route[i])
        total_distance += route[-1].distance_to_depot
    return total_distance


def mutate(solution):
    mutated_routes = [route[:] for route in solution.routes]

    # Choisir aléatoirement un client dans une route
    route_index = random.randint(0, len(mutated_routes) - 1)
    client_index = random.randint(0, len(mutated_routes[route_index]) - 1)
    client_to_move = mutated_routes[route_index].pop(client_index)

    # Trouver la meilleure position pour insérer le client dans une autre route
    best_insertion_route = None
    best_insertion_index = None
    min_distance_increase = float('inf')
    for i in range(len(mutated_routes)):
        if i != route_index:
            for j in range(len(mutated_routes[i]) + 1):
                new_route = mutated_routes[i][:j] + [client_to_move] + mutated_routes[i][j:]
                new_total_distance = calculate_total_distance(mutated_routes[:i] + [new_route] + mutated_routes[i + 1:])
                distance_increase = new_total_distance - solution.total_distance
                if distance_increase < min_distance_increase and sum(r.demand for r in new_route) <= \
                        solution.capacities[i]:
                    best_insertion_route = i
                    best_insertion_index = j
                    min_distance_increase = distance_increase

    if best_insertion_route is not None:
        mutated_routes[best_insertion_route].insert(best_insertion_index, client_to_move)
    else:
        mutated_routes[route_index].insert(client_index, client_to_move)

    return Solution(mutated_routes, solution.capacities)


# Test de l'opérateur de mutation avec plus de clients et de camions
client1 = Client(1, 10, 20)
client2 = Client(2, 15, 30)
client3 = Client(3, 20, 40)
client4 = Client(4, 25, 50)
client5 = Client(5, 12, 60)
client6 = Client(6, 18, 70)
client7 = Client(7, 22, 80)
client8 = Client(8, 28, 90)

# Créer une solution initiale avec 3 camions
routes = [[client1, client2], [client3, client4], [client5, client6, client7, client8]]
capacities = [50, 50, 75]
initial_solution = Solution(routes, capacities)

print("Solution initiale:")

print(f"Distance totale: {initial_solution.total_distance}")

# Appliquer l'opérateur de mutation
mutated_solution = mutate(initial_solution)

print("\nSolution mutée:")

print(f"Distance totale: {mutated_solution.total_distance}")