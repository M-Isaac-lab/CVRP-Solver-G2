import heapq
import random
import math
import matplotlib.pyplot as plt
import json

################ CONSTANTS #######################

MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.7
POPULATION_SIZE = 100
FITNESS = 0
TRUCKS = 1  # Nombre de camions
DEPOT = None
CAPACITY = 100
INF = float("inf")
GENERATIONS = 1000

################ CLASSES ###########################

class PrioritySet(object):
    def __init__(self):
        self.heap = []
        self.set = set()

    def push(self, d):
        if d not in self.set:
            heapq.heappush(self.heap, d)
            self.set.add(d)

    def pop(self):
        d = heapq.heappop(self.heap)
        self.set.remove(d)
        return d

    def size(self):
        return len(self.heap)

    def __getitem__(self, index):
        return self.heap[index]

class Position:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Customer:
    def __init__(self, name, x, y, demand):
        self.name = name
        self.pos = Position(x, y)
        self.demand = demand

    def __str__(self):
        return f"({self.pos.x}, {self.pos.y})"

################### UTIL FUNCTIONS ###################################

def getProb():
    return random.random()

def get_distance_matrix(cus1, cus2, distance_matrix):
    return distance_matrix[cus1.name][cus2.name]

def get_fitness(chromosome, distance_matrix):
    total_distance = 0
    current_load = 0
    routes = []
    current_route = [DEPOT]

    total_demand = sum(customer.demand for customer in chromosome)
    if total_demand > CAPACITY:
        return INF

    for customer in chromosome:
        if current_load + customer.demand <= CAPACITY:
            current_route.append(customer)
            current_load += customer.demand
        else:
            current_route.append(DEPOT)
            routes.append(current_route)
            current_route = [DEPOT, customer]
            current_load = customer.demand

    current_route.append(DEPOT)
    routes.append(current_route)

    for route in routes:
        for i in range(len(route) - 1):
            total_distance += get_distance_matrix(route[i], route[i + 1], distance_matrix)

    return total_distance if len(routes) <= TRUCKS else INF

def mutate(chromosome):
    if getProb() < MUTATION_RATE:
        left = random.randint(0, len(chromosome) - 2)
        right = random.randint(left + 1, len(chromosome) - 1)
        chromosome[left], chromosome[right] = chromosome[right], chromosome[left]
    return chromosome

def crossover(parent1, parent2):
    if getProb() < CROSSOVER_RATE:
        left = random.randint(0, len(parent1) - 2)
        right = random.randint(left + 1, len(parent1) - 1)
        child1 = parent1[:left] + parent2[left:right] + parent1[right:]
        child2 = parent2[:left] + parent1[left:right] + parent2[right:]
        return child1, child2
    return parent1, parent2

def create_population(customers):
    population = []
    for _ in range(POPULATION_SIZE):
        chromosome = random.sample(customers, len(customers))
        population.append(chromosome)
    return population

def select_parents(population):
    return random.sample(population, 2)

def plot_solution(solution, cities, filename="solution.png"):
    plt.figure(figsize=(8, 6))

    # Plot cities
    x_cities = [city.pos.x for city in cities]
    y_cities = [city.pos.y for city in cities]
    plt.scatter(x_cities, y_cities, color='blue', label='Cities')

    # Plot routes
    colors = ['red', 'green', 'purple', 'orange', 'cyan', 'brown']  # Additional colors if more than 3 trucks
    for idx, route in enumerate(solution):
        x_route = [cities[city.name].pos.x for city in route]
        y_route = [cities[city.name].pos.y for city in route]
        plt.plot(x_route, y_route, 'o-', color=colors[idx % len(colors)], label=f'Truck {idx + 1}')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Best Route Found')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def save_solution_to_file(solution, fitness, filename="solution.txt"):
    with open(filename, "w") as f:
        f.write("Solution with fitness " + str(fitness) + "\n")
        for route in solution:
            f.write("0 ")
            for c in route:
                f.write(str(c.name) + " ")
            f.write("0\n")

##################### EVOLUTION FUNCTION ##############################

def Genetic_Algo(customers, distance_matrix):
    population = create_population(customers)
    best_chromosome = None
    best_fitness = INF

    for generation in range(GENERATIONS):
        new_population = []
        for _ in range(POPULATION_SIZE // 2):
            parent1, parent2 = select_parents(population)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)
            new_population.extend([child1, child2])

        population = sorted(new_population, key=lambda x: get_fitness(x, distance_matrix))[:POPULATION_SIZE]
        current_best = population[0]
        current_fitness = get_fitness(current_best, distance_matrix)

        if current_fitness < best_fitness:
            best_chromosome = current_best
            best_fitness = current_fitness
            print(f"Generation {generation}: Best Fitness = {best_fitness}")

    best_routes = split_into_routes(best_chromosome)
    return best_routes, best_fitness

def split_into_routes(chromosome):
    routes = []
    current_route = [DEPOT]
    current_load = 0

    for customer in chromosome:
        if current_load + customer.demand <= CAPACITY:
            current_route.append(customer)
            current_load += customer.demand
        else:
            current_route.append(DEPOT)
            routes.append(current_route)
            current_route = [DEPOT, customer]
            current_load = customer.demand

    current_route.append(DEPOT)
    routes.append(current_route)

    return routes

######################## DATA ########################################

def create_data_array(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
        cities = data['cities']
        distance_matrix = data['distances']
        demands = data['demandes']
        capacities = data['trucks_capacities']
        
    customers = [Customer(i, cities[i][0], cities[i][1], demands[i]) for i in range(len(cities))]
    
    global DEPOT, CAPACITY
    DEPOT = Customer(0, cities[0][0], cities[0][1], demands[0])
    CAPACITY = capacities[0]

    return customers, distance_matrix, cities

##################### MAIN ###########################################

if __name__ == "__main__":
    filename = 'graphes/14nodes_0dindex_1trucks.json.json'
    customers, distance_matrix, cities = create_data_array(filename)
    best_routes, best_fitness = Genetic_Algo(customers, distance_matrix)
    if best_routes is not None:
        save_solution_to_file(best_routes, best_fitness)
        plot_solution(best_routes, customers)
        print("Best solution found with fitness:", best_fitness)
