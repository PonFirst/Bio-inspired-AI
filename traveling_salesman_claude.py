import numpy as np
import pygad
import matplotlib.pyplot as plt

# Define the cities (x, y coordinates)
# cities = np.array([
#     [60, 200], [180, 200], [80, 180], [140, 180], [20, 160], 
#     [100, 160], [200, 160], [140, 140], [40, 120], [100, 120], 
#     [180, 100], [60, 80], [120, 80], [180, 60], [20, 40], 
#     [100, 40], [200, 40], [20, 20], [60, 20], [160, 20],
#     [240, 190], [30, 190], [170, 170], [90, 150], [210, 150],
#     [50, 130], [130, 130], [220, 110], [70, 110], [150, 90],
#     [190, 90], [40, 70], [110, 70], [160, 70], [200, 50],
#     [30, 50], [80, 30], [140, 30], [190, 30], [220, 180],
#     [10, 180], [120, 200], [160, 190], [50, 170], [110, 170],
#     [180, 150], [90, 140], [150, 120], [210, 100], [250, 80]
# ])

num_cities = 50
np.random.seed(42)
cities = np.random.rand(num_cities, 2) * 100

# num_cities = len(cities)

def calculate_distance(city1, city2):
    """Calculate Euclidean distance between two cities"""
    return np.sqrt(np.sum((city1 - city2) ** 2))

def create_distance_matrix():
    """Create distance matrix for all city pairs"""
    distance_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                distance_matrix[i][j] = calculate_distance(cities[i], cities[j])
    return distance_matrix

# Create distance matrix
distance_matrix = create_distance_matrix()

def fitness_function(ga_instance, solution, solution_idx):
    """
    Calculate fitness for a TSP solution
    Fitness = 1 / total_distance (higher fitness = shorter distance)
    """
    total_distance = 0
    
    # Convert solution to integer indices
    route = solution.astype(int)
    
    # Calculate total distance for the route
    for i in range(len(route)):
        current_city = route[i]
        next_city = route[(i + 1) % len(route)]  # Return to start city
        total_distance += distance_matrix[current_city][next_city]
    
    # Return fitness (inverse of distance)
    return 1.0 / total_distance if total_distance > 0 else 0

def on_generation(ga_instance):
    """Callback function to track progress"""
    generation = ga_instance.generations_completed
    fitness = ga_instance.best_solution()[1]
    best_distance = 1.0 / fitness if fitness > 0 else float('inf')
    
    print(f"Generation {generation}: Best Distance = {best_distance:.2f}")

# Define the gene space (each gene can be any city index from 0 to num_cities-1)
gene_space = list(range(num_cities))

# GA parameters
ga_instance = pygad.GA(
    num_generations=10000,
    num_parents_mating=10,
    fitness_func=fitness_function,
    sol_per_pop=80,
    num_genes=num_cities,
    gene_space=gene_space,
    gene_type=int,
    parent_selection_type="sss",
    keep_parents=5,
    crossover_type="single_point",
    mutation_type="inversion",
    mutation_percent_genes=20,
    allow_duplicate_genes=False,
    on_generation=on_generation,
    random_seed=42
)

print("Starting Genetic Algorithm for TSP...")
print(f"Number of cities: {num_cities}")
print(f"Population size: {ga_instance.sol_per_pop}")
print(f"Number of generations: {ga_instance.num_generations}")
print("-" * 50)

# Run the GA
ga_instance.run()

print("-" * 50)
print("GA completed!")

# Get the best solution
solution, solution_fitness, solution_idx = ga_instance.best_solution()
best_route = solution.astype(int)
best_distance = 1.0 / solution_fitness

print(f"Best route: {best_route}")
print(f"Best distance: {best_distance:.2f}")
print(f"Best fitness: {solution_fitness:.6f}")

# Plot the fitness evolution
plt.figure(figsize=(12, 5))

# Plot 1: Fitness over generations
plt.subplot(1, 2, 1)
plt.plot(ga_instance.best_solutions_fitness)
plt.title('Best Fitness over Generations')
plt.xlabel('Generation')
plt.ylabel('Fitness (1/Distance)')
plt.grid(True)

# Plot 2: Distance over generations (more intuitive)
distances = [1.0/fitness if fitness > 0 else float('inf') for fitness in ga_instance.best_solutions_fitness]
plt.subplot(1, 2, 2)
plt.plot(distances)
plt.title('Best Distance over Generations')
plt.xlabel('Generation')
plt.ylabel('Total Distance')
plt.grid(True)

plt.tight_layout()
plt.show()

# Visualize the best route
plt.figure(figsize=(10, 8))

# Plot cities
plt.scatter(cities[:, 0], cities[:, 1], c='red', s=100, alpha=0.7, zorder=5)

# Plot the best route
route_cities = cities[best_route]
route_x = np.append(route_cities[:, 0], route_cities[0, 0])  # Return to start
route_y = np.append(route_cities[:, 1], route_cities[0, 1])  # Return to start

plt.plot(route_x, route_y, 'b-', linewidth=2, alpha=0.7)
plt.plot(route_x, route_y, 'bo', markersize=8, alpha=0.7)

# Label cities
for i, (x, y) in enumerate(cities):
    plt.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points', fontsize=10)

plt.title(f'Best TSP Route (Distance: {best_distance:.2f})')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.show()

print(f"\nFinal Results:")
print(f"Best route found: {' -> '.join(map(str, best_route))} -> {best_route[0]}")
print(f"Total distance: {best_distance:.2f}")