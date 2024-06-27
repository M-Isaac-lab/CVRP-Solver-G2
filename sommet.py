import heapq
import random
import math
import matplotlib.pyplot as plt

################ CONSTANTS #######################

MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.7
POPULATION_SIZE = 100
FITNESS = 0
TRUCKS = 6
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

def get_distance(cus1, cus2):
    return math.sqrt((cus1.pos.x - cus2.pos.x) ** 2 + (cus1.pos.y - cus2.pos.y) ** 2)

def get_fitness(chromosome):
    total_distance = 0
    current_load = 0
    routes = []
    current_route = [DEPOT]

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
            total_distance += get_distance(route[i], route[i + 1])

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

def plot_solution(solution, filename="solution.png"):
    plt.figure()
    for route in solution:
        x = [c.pos.x for c in route]
        y = [c.pos.y for c in route]
        plt.plot(x, y, 'o-')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Best Route Found')
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

def Genetic_Algo(customers):
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

        population = sorted(new_population, key=get_fitness)[:POPULATION_SIZE]
        current_best = population[0]
        current_fitness = get_fitness(current_best)

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

def create_data_array():
    locations = [(15, 19), (1, 49), (87, 25), (69, 65), (93, 91), (33, 31), (71, 61), (29, 9), (93, 7),
                 (55, 47), (23, 13), (19, 47), (57, 63), (5, 95), (65, 43), (69, 1), (3, 25), (19, 91),
                 (21, 81), (67, 91), (41, 23), (19, 75), (15, 79), (79, 47), (19, 65), (27, 49), (29, 17),
                 (25, 65), (59, 51), (27, 95), (21, 91), (61, 83), (15, 83), (31, 87), (71, 41), (91, 21)]
    
    demands = [0, 1, 14, 15, 11, 18, 2, 22, 7, 18, 23, 12, 21, 2, 14, 9, 10, 4, 19, 2, 20, 15,
               11, 6, 13, 19, 13, 8, 15, 18, 11, 21, 12, 2, 23, 11]

    customers = [Customer(i, locations[i][0], locations[i][1], demands[i]) for i in range(1, len(locations))]
    
    global DEPOT
    DEPOT = Customer(0, locations[0][0], locations[0][1], demands[0])

    return customers

##################### MAIN ###########################################

if __name__ == "__main__":
    customers = create_data_array()
    best_routes, best_fitness = Genetic_Algo(customers)
    if best_routes is not None:
        save_solution_to_file(best_routes, best_fitness)
        plot_solution(best_routes)
        print("Best solution found with fitness:", best_fitness)
