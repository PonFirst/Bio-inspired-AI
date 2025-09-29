import numpy as np
import matplotlib.pyplot as plt
from mealpy.swarm_based import ABC
from mealpy import FloatVar


# ---- Berlin52 dataset ----
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

# ---- Distance matrix ----
def calculate_distance(city1, city2):
    return np.linalg.norm(city1 - city2)

def create_distance_matrix():
    distance_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                distance_matrix[i][j] = calculate_distance(cities[i], cities[j])
    return distance_matrix

distance_matrix = create_distance_matrix()

# ---- Improved fitness function with 2-opt local search ----
def two_opt(route, distance_matrix):
    """Apply 2-opt local search to improve the route"""
    improved = True
    best_route = route.copy()
    best_distance = calculate_route_distance(best_route, distance_matrix)
    
    while improved:
        improved = False
        for i in range(1, len(route) - 1):
            for j in range(i + 1, len(route)):
                # Reverse the segment between i and j
                new_route = best_route.copy()
                new_route[i:j+1] = new_route[i:j+1][::-1]
                new_distance = calculate_route_distance(new_route, distance_matrix)
                
                if new_distance < best_distance:
                    best_route = new_route
                    best_distance = new_distance
                    improved = True
                    break
            if improved:
                break
    
    return best_route, best_distance

def calculate_route_distance(route, distance_matrix):
    """Calculate total distance of a route"""
    total_distance = 0
    for i in range(len(route)):
        current_city = route[i]
        next_city = route[(i + 1) % len(route)]
        total_distance += distance_matrix[current_city][next_city]
    return total_distance

def tsp_fitness(solution):
    """Enhanced fitness function with multiple sampling and local optimization"""
    # Generate multiple candidate routes from the continuous solution
    # Use different ranking methods to create diversity
    best_distance = float('inf')
    
    # Method 1: Standard argsort
    route1 = np.argsort(solution)
    dist1 = calculate_route_distance(route1, distance_matrix)
    if dist1 < best_distance:
        best_distance = dist1
    
    # Method 2: Add random noise and argsort (exploration)
    noise = np.random.uniform(-0.01, 0.01, len(solution))
    route2 = np.argsort(solution + noise)
    dist2 = calculate_route_distance(route2, distance_matrix)
    if dist2 < best_distance:
        best_distance = dist2
    
    # Method 3: Inverse ranking
    route3 = np.argsort(-solution)
    dist3 = calculate_route_distance(route3, distance_matrix)
    if dist3 < best_distance:
        best_distance = dist3
    
    return best_distance

# ---- Problem definition for mealpy ----
problem = {
    "obj_func": tsp_fitness,
    "bounds": FloatVar(lb=[0]*num_cities, ub=[10]*num_cities),  # Larger range for better diversity
    "minmax": "min",
}

# ---- Run ABC with better parameters ----
print("Running ABC optimization...")
model = ABC.OriginalABC(
    epoch=10000,      # More iterations
    pop_size=100,     # Larger population
    n_limits=50,      # Limit for abandonment (balance exploration/exploitation)
)
best_agent = model.solve(problem)

# Extract solution from the agent
best_position = best_agent.solution
best_distance = best_agent.target.fitness

# Decode best solution and apply 2-opt improvement
best_route = np.argsort(best_position)
print(f"\nInitial best distance: {best_distance:.2f}")

# Apply 2-opt local search for final improvement
print("Applying 2-opt local search...")
best_route, best_distance = two_opt(best_route, distance_matrix)

print(f"Best route after 2-opt: {best_route}")
print(f"Final best distance: {best_distance:.2f}")
print(f"\nNote: Optimal Berlin52 distance is 7542")

# ---- Plot best route ----
plt.figure(figsize=(12, 10))
plt.scatter(cities[:, 0], cities[:, 1], c='red', s=150, alpha=0.8, zorder=5, edgecolors='black', linewidths=2)

route_cities = cities[best_route]
route_x = np.append(route_cities[:, 0], route_cities[0, 0])
route_y = np.append(route_cities[:, 1], route_cities[0, 1])

plt.plot(route_x, route_y, 'b-', linewidth=2.5, alpha=0.7)
plt.plot(route_x, route_y, 'bo', markersize=10, alpha=0.5)

# Mark start city
plt.scatter(cities[best_route[0], 0], cities[best_route[0], 1], 
           c='green', s=300, marker='*', zorder=10, edgecolors='black', linewidths=2, label='Start')

for i, (x, y) in enumerate(cities):
    plt.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points', 
                fontsize=9, fontweight='bold')

plt.title(f'Best Berlin52 Route (ABC + 2-opt)\nDistance: {best_distance:.2f} (Optimal: 7542)', 
         fontsize=14, fontweight='bold')
plt.xlabel('X Coordinate', fontsize=12)
plt.ylabel('Y Coordinate', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.axis('equal')
plt.tight_layout()
plt.show()