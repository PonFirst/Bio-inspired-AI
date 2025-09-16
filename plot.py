import numpy as np
import matplotlib.pyplot as plt

# Define the cities (x, y coordinates)
cities = np.array([
    [60, 200], [180, 200], [80, 180], [140, 180], [20, 160], 
    [100, 160], [200, 160], [140, 140], [40, 120], [100, 120], 
    [180, 100], [60, 80], [120, 80], [180, 60], [20, 40], 
    [100, 40], [200, 40], [20, 20], [60, 20], [160, 20],
    [240, 190], [30, 190], [170, 170], [90, 150], [210, 150],
    [50, 130], [130, 130], [220, 110], [70, 110], [150, 90],
    [190, 90], [40, 70], [110, 70], [160, 70], [200, 50],
    [30, 50], [80, 30], [140, 30], [190, 30], [220, 180],
    [10, 180], [120, 200], [160, 190], [50, 170], [110, 170],
    [180, 150], [90, 140], [150, 120], [210, 100], [250, 80]
])

num_cities = len(cities)

# Plot the cities
plt.figure(figsize=(12, 10))
plt.scatter(cities[:, 0], cities[:, 1], c='red', s=100, alpha=0.7, zorder=5)

# Label cities with their index numbers
for i, (x, y) in enumerate(cities):
    plt.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points', fontsize=10)

plt.title(f'TSP Problem: {num_cities} Cities (Starting Point)', fontsize=16)
plt.xlabel('X Coordinate', fontsize=12)
plt.ylabel('Y Coordinate', fontsize=12)
plt.grid(True, alpha=0.3)
plt.axis('equal')

# Add some padding to the plot
x_margin = (cities[:, 0].max() - cities[:, 0].min()) * 0.1
y_margin = (cities[:, 1].max() - cities[:, 1].min()) * 0.1
plt.xlim(cities[:, 0].min() - x_margin, cities[:, 0].max() + x_margin)
plt.ylim(cities[:, 1].min() - y_margin, cities[:, 1].max() + y_margin)

plt.tight_layout()
plt.show()

print(f"Total number of cities: {num_cities}")
print(f"X coordinates range: {cities[:, 0].min()} to {cities[:, 0].max()}")
print(f"Y coordinates range: {cities[:, 1].min()} to {cities[:, 1].max()}")