import numpy as np
import matplotlib.pyplot as plt
import pygad

# -------------------
# STEP 1: Berlin52 Dataset
# -------------------
cities = [
    (565, 575), (25, 185), (345, 750), (945, 685), (845, 655), (880, 660), (25, 230),
    (525, 1000), (580, 1175), (650, 1130), (1605, 620), (1220, 580), (1465, 200), (1530, 5),
    (845, 680), (725, 370), (145, 665), (415, 635), (510, 875), (560, 365), (300, 465),
    (520, 585), (480, 415), (835, 625), (975, 580), (1215, 245), (1320, 315), (1250, 400),
    (660, 180), (410, 250), (420, 555), (575, 665), (1150, 1160), (700, 580), (685, 595),
    (685, 610), (770, 610), (795, 645), (720, 635), (760, 650), (475, 960), (95, 260),
    (875, 920), (700, 500), (555, 815), (830, 485), (1170, 65), (830, 610), (605, 625),
    (595, 360), (1340, 725), (1740, 245)
]

num_cities = len(cities)

def calculate_distance(city1, city2):
    return np.linalg.norm(np.array(city1) - np.array(city2))

# Distance matrix
distance_matrix = np.zeros((num_cities, num_cities))
for i in range(num_cities):
    for j in range(num_cities):
        if i != j:
            distance_matrix[i][j] = calculate_distance(cities[i], cities[j])

# -------------------
# STEP 2: Ant Colony Optimization Implementation
# -------------------
class AntColony:
    def __init__(self, nodes, ant_count, alpha, beta, pheromone_evaporation_rate, pheromone_constant, iterations):
        self.nodes = nodes
        self.num_nodes = len(nodes)
        self.ant_count = ant_count
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = pheromone_evaporation_rate
        self.pheromone_constant = pheromone_constant
        self.iterations = iterations
        self.pheromone = np.ones((self.num_nodes, self.num_nodes)) / self.num_nodes
        self.distances = distance_matrix

    def get_path(self):
        best_path = None
        best_length = float('inf')
        for _ in range(self.iterations):
            paths = []
            path_lengths = []
            for _ in range(self.ant_count):
                path = self._construct_path()
                path_length = self._calculate_path_length(path)
                paths.append(path)
                path_lengths.append(path_length)
                if path_length < best_length:
                    best_length = path_length
                    best_path = path
            self._update_pheromones(paths, path_lengths)
        # Convert indices to coordinates
        return [self.nodes[i] for i in best_path]

    def _construct_path(self):
        path = [np.random.randint(self.num_nodes)]
        unvisited = set(range(self.num_nodes)) - {path[0]}
        while unvisited:
            next_node = self._choose_next_node(path[-1], unvisited)
            path.append(next_node)
            unvisited.remove(next_node)
        return path

    def _choose_next_node(self, current, unvisited):
        probabilities = []
        total = 0
        for next_node in unvisited:
            if self.distances[current][next_node] == 0:
                prob = 0
            else:
                prob = (self.pheromone[current][next_node] ** self.alpha) * \
                       ((1.0 / self.distances[current][next_node]) ** self.beta)
            probabilities.append(prob)
            total += prob
        if total == 0:
            return np.random.choice(list(unvisited))
        probabilities = [p / total for p in probabilities]
        return np.random.choice(list(unvisited), p=probabilities)

    def _calculate_path_length(self, path):
        length = 0
        for i in range(len(path)):
            length += self.distances[path[i]][path[(i + 1) % len(path)]]
        return length

    def _update_pheromones(self, paths, path_lengths):
        self.pheromone *= (1 - self.evaporation_rate)
        for path, length in zip(paths, path_lengths):
            contribution = self.pheromone_constant / length
            for i in range(len(path)):
                self.pheromone[path[i]][path[(i + 1) % len(path)]] += contribution

# -------------------
# STEP 3: 2-opt Refinement Function
# -------------------
def two_opt(route, distance_matrix):
    n = len(route)
    improved = True
    best_route = route.copy()
    best_distance = sum(distance_matrix[best_route[i]][best_route[(i+1)%n]] for i in range(n))
    while improved:
        improved = False
        for i in range(1, n-2):
            for j in range(i+1, n):
                if j - i == 1: continue
                old_dist = (distance_matrix[best_route[i-1]][best_route[i]] +
                           distance_matrix[best_route[j]][best_route[(j+1)%n]])
                new_dist = (distance_matrix[best_route[i-1]][best_route[j]] +
                           distance_matrix[best_route[i]][best_route[(j+1)%n]])
                if new_dist < old_dist:
                    best_route[i:j+1] = best_route[i:j+1][::-1]
                    improved = True
                    best_distance = sum(distance_matrix[best_route[k]][best_route[(k+1)%n]] for k in range(n))
    return best_route, best_distance

# -------------------
# STEP 4: GA Crossover Function (Order Crossover)
# -------------------
def order_crossover(parents, offspring_size, ga_instance):
    num_offspring, size = offspring_size
    offspring = np.empty((num_offspring, size), dtype=int)
    for k in range(num_offspring):
        parent1_idx = k % parents.shape[0]
        parent2_idx = (k + 1) % parents.shape[0]
        parent1 = parents[parent1_idx].copy()
        parent2 = parents[parent2_idx].copy()
        child = -np.ones(size, dtype=int)
        start, end = sorted(np.random.choice(range(size), 2, replace=False))
        child[start:end] = parent1[start:end]
        ptr = end
        for gene in np.concatenate((parent2[end:], parent2[:end])):
            if gene not in child:
                child[ptr % size] = gene
                ptr += 1
        offspring[k, :] = child
    return offspring

# -------------------
# MAIN PIPELINE
# -------------------
def main():
    print("Running ACO optimization...")

    # Run ACO
    aco = AntColony(
        nodes=cities,
        ant_count=50,
        alpha=1.0,
        beta=3.0,
        pheromone_evaporation_rate=0.5,
        pheromone_constant=1000.0,
        iterations=200
    )

    best_path_coords = aco.get_path()
    best_path = [list(cities).index(coord) for coord in best_path_coords]
    best_distance = sum(distance_matrix[best_path[i]][best_path[(i + 1) % len(best_path)]] for i in range(len(best_path)))
    print(f"[ACO] Best path length: {best_distance:.2f}")

    # Apply 2-opt to ACO solution
    aco_refined_path, aco_refined_distance = two_opt(best_path, distance_matrix)
    print(f"[ACO + 2-opt] Refined path length: {aco_refined_distance:.2f}")

    # GA Fitness Function
    def fitness_func(ga_instance, solution, solution_idx):
        route = solution.astype(int)
        total_distance = sum(distance_matrix[route[i]][route[(i+1)%len(route)]] for i in range(len(route)))
        return 1.0 / total_distance if total_distance > 0 else 0

    # Seed GA with ACO result
    initial_population = []
    initial_population.append(aco_refined_path)  # Exact ACO solution
    for _ in range(30):
        perm = aco_refined_path.copy()
        np.random.shuffle(perm)
        initial_population.append(perm)
    for _ in range(19):
        initial_population.append(np.random.permutation(num_cities))
    initial_population = np.array(initial_population)

    # Run GA
    print("\nRunning Genetic Algorithm...")
    ga_instance = pygad.GA(
        num_generations=500,
        num_parents_mating=10,
        fitness_func=fitness_func,
        sol_per_pop=50,
        num_genes=num_cities,
        gene_space=list(range(num_cities)),
        gene_type=int,
        parent_selection_type="sss",
        keep_elitism=2,
        crossover_type=order_crossover,
        mutation_type="inversion",
        mutation_percent_genes=15,
        allow_duplicate_genes=False,
        initial_population=initial_population,
        on_generation=lambda ga: (
            print(f"GA Generation {ga.generations_completed}: Best distance = {1.0/ga.best_solution()[1]:.2f}")
            if ga.generations_completed % 50 == 0 else None
        ),
        random_seed=42
    )

    ga_instance.run()

    solution, solution_fitness, _ = ga_instance.best_solution()
    hybrid_route = solution.astype(int)
    hybrid_distance = 1.0 / solution_fitness
    print(f"[Hybrid ACO+GA] Best path length: {hybrid_distance:.2f}")

    # Apply 2-opt to GA solution
    hybrid_refined_route, hybrid_refined_distance = two_opt(hybrid_route, distance_matrix)
    print(f"[Hybrid ACO+GA + 2-opt] Refined path length: {hybrid_refined_distance:.2f}")

    # Plot Hybrid Route
    cities_array = np.array(cities)
    plt.figure(figsize=(10, 8))
    plt.scatter(cities_array[:, 0], cities_array[:, 1], c='red', s=100, alpha=0.7, zorder=5)
    route_cities = cities_array[hybrid_refined_route]
    route_x = np.append(route_cities[:, 0], route_cities[0, 0])
    route_y = np.append(route_cities[:, 1], route_cities[0, 1])
    plt.plot(route_x, route_y, 'b-', linewidth=2, alpha=0.7)
    plt.plot(route_x, route_y, 'bo', markersize=8, alpha=0.7)

    for i, (x, y) in enumerate(cities):
        plt.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points', fontsize=10)

    plt.title(f'Hybrid ACO+GA+2-opt TSP (Distance: {hybrid_refined_distance:.2f})')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()

if __name__ == '__main__':
    main()