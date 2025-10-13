import numpy as np
import random
import matplotlib.pyplot as plt

# --- Load Berlin52 coordinates ---
cities = np.array([
    (565, 575), (25, 185), (345, 750), (945, 685), (845, 655), (880, 660), (25, 230),
    (525, 1000), (580, 1175), (650, 1130), (1605, 620), (1220, 580), (1465, 200), (1530, 5),
    (845, 680), (725, 370), (145, 665), (415, 635), (510, 875), (560, 365), (300, 465),
    (520, 585), (480, 415), (835, 625), (975, 580), (1215, 245), (1320, 315), (1250, 400),
    (660, 180), (410, 250), (420, 555), (575, 665), (1150, 1160), (700, 580), (685, 595),
    (685, 610), (770, 610), (795, 645), (720, 635), (760, 650), (475, 960), (95, 260),
    (875, 920), (700, 500), (555, 815), (830, 485), (1170, 65), (830, 610), (605, 625),
    (595, 360), (1340, 725), (1740, 245)
])
num_cities = len(cities)

# --- Distance matrix ---
distance_matrix = np.zeros((num_cities, num_cities))
for i in range(num_cities):
    for j in range(num_cities):
        if i != j:
            distance_matrix[i][j] = np.linalg.norm(cities[i] - cities[j])

# --- Utility: Compute tour distance ---
def tour_length(route):
    return sum(distance_matrix[route[i], route[(i+1) % len(route)]] for i in range(len(route)))

# --- Mutation operator (swap or inversion) ---
def mutate(route, rate=0.1):
    r = route.copy()
    if random.random() < 0.5:
        # Swap mutation
        for _ in range(int(rate * len(route))):
            i, j = random.sample(range(len(route)), 2)
            r[i], r[j] = r[j], r[i]
    else:
        # Inversion mutation
        i, j = sorted(random.sample(range(len(route)), 2))
        r[i:j] = list(reversed(r[i:j]))
    return r


# --- CSA for TSP ---
def clonal_selection_tsp(generations=500, pop_size=80, elite_size=10,
                         clone_factor=5, mutation_base=0.3, newcomers=15, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    # Initialize random population (list of city index permutations)
    population = [np.random.permutation(num_cities) for _ in range(pop_size)]
    best_route, best_len = None, float('inf')
    history = []

    for gen in range(generations):
        # Evaluate fitness (shorter = better)
        fitness = np.array([tour_length(r) for r in population])
        order = np.argsort(fitness)
        population = [population[i] for i in order]
        fitness = fitness[order]

        # Update global best
        if fitness[0] < best_len:
            best_len = fitness[0]
            best_route = population[0].copy()

        history.append(best_len)

        # Select elite antibodies
        elite = population[:elite_size]
        elite_fitness = fitness[:elite_size]
        maxL, minL = elite_fitness.max(), elite_fitness.min()
        affinity = 1 - (elite_fitness - minL) / (maxL - minL + 1e-9)

        # Clone proportionally to affinity
        clones = []
        for i, ab in enumerate(elite):
            n_clones = max(1, int(clone_factor * (1 + affinity[i])))
            clones += [ab.copy() for _ in range(n_clones)]

        # Mutate clones with inverse relation to affinity
        mutated = []
        for i, cl in enumerate(clones):
            rate = mutation_base * (1 - affinity[i % len(affinity)])
            mutated.append(mutate(cl, rate))

        # New random antibodies for diversity
        new_random = [np.random.permutation(num_cities) for _ in range(newcomers)]

        # Combine and select next generation
        population = elite + mutated + new_random
        population = sorted(population, key=lambda r: tour_length(r))[:pop_size]

        # Progress print
        if gen % 25 == 0:
            print(f"Generation {gen}/{generations}: Best distance = {best_len:.2f}")

    return best_route, best_len, history

# --- Run CSA ---
best_route, best_len, history = clonal_selection_tsp(generations=500)
print(f"\nBest distance found: {best_len:.2f}")
print(f"Best route: {best_route}")

# --- Plot convergence ---
plt.figure(figsize=(10, 6))
plt.plot(history, color='darkorange', linewidth=2)
plt.title("Clonal Selection Algorithm (CSA) on Berlin52")
plt.xlabel("Generation")
plt.ylabel("Best Distance")
plt.grid(True, alpha=0.3)
plt.show()

# --- Plot best route ---
plt.figure(figsize=(10, 8))
plt.scatter(cities[:, 0], cities[:, 1], c='red', s=100, zorder=5)
route_coords = cities[best_route]
plt.plot(np.append(route_coords[:, 0], route_coords[0, 0]),
         np.append(route_coords[:, 1], route_coords[0, 1]),
         'b-', linewidth=2, alpha=0.75, label='Best Route')

for i, (x, y) in enumerate(cities):
    plt.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)

plt.title(f"Best Route Found (Distance: {best_len:.2f})", fontsize=14)
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.legend()
plt.show()
