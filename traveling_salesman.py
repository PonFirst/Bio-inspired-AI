import numpy as np
import pygad
import time
import matplotlib.pyplot as plt

# Example: 20 cities with random coordinates
num_cities = 20
np.random.seed(42) # For reproducibility 
city_coords = np.random.rand(num_cities, 2) * 100  # (x, y) between 0–100

# Distance matrix
dist_matrix = np.sqrt(((city_coords[:, None, :] - city_coords[None, :, :])**2).sum(axis=2))

# Fitness function = negative tour length (since GA maximizes)
def fitness_func(ga, solution, sol_idx):
    tour_length = 0
    for i in range(len(solution) - 1):
        tour_length += dist_matrix[int(solution[i]), int(solution[i+1])]
    # Return to start
    tour_length += dist_matrix[int(solution[-1]), int(solution[0])]
    return -tour_length

# Gene space = indices of cities
gene_space = list(range(num_cities))

ga = pygad.GA(
    num_generations    = 1000,
    num_parents_mating = 20,
    fitness_func       = fitness_func,
    sol_per_pop        = 50,
    num_genes          = num_cities,
    gene_space         = gene_space,
    parent_selection_type   = "tournament",
    keep_parents           = 5,
    crossover_type         = "single_point",
    mutation_type          = "random",
    mutation_percent_genes = 20,
    allow_duplicate_genes  = False  # Important for permutations!
)

start = time.time()
ga.run()
elapsed = time.time() - start

best_solution, best_fitness, _ = ga.best_solution()
best_cost = -best_fitness
iterations = ga.generations_completed

print(f"Time Passed: {elapsed:.6f} seconds")
print(f"Best Tour Cost: {best_cost:.6f}")
print(f"Best Tour Order: {best_solution}")
print(f"Iterations: {iterations}")

# ---- Plot the best tour ----
ordered_coords = city_coords[best_solution]
# Add the return to start
ordered_coords = np.vstack([ordered_coords, ordered_coords[0]])

plt.figure(figsize=(8, 6))
plt.plot(ordered_coords[:, 0], ordered_coords[:, 1], '-o', color="blue", markersize=8)
plt.scatter(city_coords[:, 0], city_coords[:, 1], c="red", s=50, label="Cities")
for idx, (x, y) in enumerate(city_coords):
    plt.text(x+1, y+1, str(idx), fontsize=9)

plt.title(f"TSP Best Route (Cost = {best_cost:.2f})")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.show()
