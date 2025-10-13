import numpy as np
import random
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

# --- Berlin52 setup ---
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
distance_matrix = np.linalg.norm(cities[:, None] - cities, axis=2)

def tour_length(route):
    return sum(distance_matrix[route[i], route[(i+1)%len(route)]] for i in range(len(route)))

# --- Mutation operator ---
def mutate(route, rate=0.1):
    r = route.copy()
    if random.random() < 0.5:
        for _ in range(int(rate * len(route))):
            i, j = random.sample(range(len(route)), 2)
            r[i], r[j] = r[j], r[i]
    else:
        i, j = sorted(random.sample(range(len(route)), 2))
        r[i:j] = list(reversed(r[i:j]))
    return r

# --- CSA Core ---
def clonal_selection_tsp(generations=500, pop_size=80, elite_size=10,
                         clone_factor=5, mutation_base=0.3, newcomers=15, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    population = [np.random.permutation(num_cities) for _ in range(pop_size)]
    best_route, best_len = None, float('inf')
    history = []

    for gen in range(generations):
        fitness = np.array([tour_length(r) for r in population])
        order = np.argsort(fitness)
        population = [population[i] for i in order]
        fitness = fitness[order]

        if fitness[0] < best_len:
            best_len = fitness[0]
            best_route = population[0].copy()

        history.append(best_len)

        elite = population[:elite_size]
        elite_fitness = fitness[:elite_size]
        maxL, minL = elite_fitness.max(), elite_fitness.min()
        affinity = 1 - (elite_fitness - minL) / (maxL - minL + 1e-9)

        clones = []
        for i, ab in enumerate(elite):
            n_clones = max(1, int(clone_factor * (1 + affinity[i])))
            clones += [ab.copy() for _ in range(n_clones)]

        mutated = []
        for i, cl in enumerate(clones):
            rate = mutation_base * (1 - affinity[i % len(affinity)])
            mutated.append(mutate(cl, rate))

        new_random = [np.random.permutation(num_cities) for _ in range(newcomers)]
        population = elite + mutated + new_random
        population = sorted(population, key=lambda r: tour_length(r))[:pop_size]

    return best_route, best_len, history


# --- GRID SEARCH + MLFLOW ---
param_grid = {
    "pop_size": [60, 80, 100],
    "elite_size": [8, 10, 15],
    "clone_factor": [4, 6, 8],
    "mutation_base": [0.2, 0.3, 0.4],
    "newcomers": [10, 20]
}

mlflow.set_experiment("ClonalSelection_Berlin52")

best_overall = float('inf')
best_params = None

for pop_size in param_grid["pop_size"]:
    for elite_size in param_grid["elite_size"]:
        for clone_factor in param_grid["clone_factor"]:
            for mutation_base in param_grid["mutation_base"]:
                for newcomers in param_grid["newcomers"]:
                    with mlflow.start_run():
                        params = {
                            "pop_size": pop_size,
                            "elite_size": elite_size,
                            "clone_factor": clone_factor,
                            "mutation_base": mutation_base,
                            "newcomers": newcomers
                        }
                        mlflow.log_params(params)
                        best_route, best_len, history = clonal_selection_tsp(
                            generations=300,
                            pop_size=pop_size,
                            elite_size=elite_size,
                            clone_factor=clone_factor,
                            mutation_base=mutation_base,
                            newcomers=newcomers
                        )
                        mlflow.log_metric("best_distance", best_len)

                        if best_len < best_overall:
                            best_overall = best_len
                            best_params = params

                        

# === After the Grid Search Loop ===
print("\n=== GRID SEARCH RESULTS ===")
print(f"Best distance: {best_overall:.2f}")
print(f"Best parameters: {best_params}")

# --- Plot convergence curve for best run ---
print("\nRe-running best configuration to visualize convergence...")
best_route, best_len, history = clonal_selection_tsp(
    generations=300,
    pop_size=best_params["pop_size"],
    elite_size=best_params["elite_size"],
    clone_factor=best_params["clone_factor"],
    mutation_base=best_params["mutation_base"],
    newcomers=best_params["newcomers"]
)

plt.figure(figsize=(8, 5))
plt.plot(history, color='darkorange', linewidth=2)
plt.title(f"Best AIS using Clonal Selection Convergence (Distance: {best_len:.2f})")
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