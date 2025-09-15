import numpy as np
import pygad
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider

# ---- Setup problem ----
num_cities = 100
np.random.seed(42)
city_coords = np.random.rand(num_cities, 2) * 100  # random cities
dist_matrix = np.sqrt(((city_coords[:, None, :] - city_coords[None, :, :])**2).sum(axis=2))

def fitness_func(ga, solution, sol_idx):
    tour_length = 0
    for i in range(len(solution) - 1):
        tour_length += dist_matrix[int(solution[i]), int(solution[i+1])]
    tour_length += dist_matrix[int(solution[-1]), int(solution[0])]
    return -tour_length

gene_space = list(range(num_cities))

# store history = (tour, cost, generation)
history = []

def on_generation(ga):
    best_solution, best_fitness, _ = ga.best_solution()
    best_solution = np.array(best_solution, dtype=int)
    best_cost = -best_fitness
    history.append((best_solution, best_cost, ga.generations_completed))

ga = pygad.GA(
    num_generations=200,
    num_parents_mating=20,
    fitness_func=fitness_func,
    sol_per_pop=50,
    num_genes=num_cities,
    gene_space=gene_space,
    parent_selection_type="tournament",
    keep_parents=5,
    crossover_type="single_point",
    mutation_type="random",
    mutation_percent_genes=20,
    allow_duplicate_genes=False
)

ga.on_generation = on_generation
start = time.time()
ga.run()
elapsed = time.time() - start

best_solution, best_fitness, _ = ga.best_solution()
best_cost = -best_fitness
print(f"Time Passed: {elapsed:.6f} seconds")
print(f"Best Tour Cost: {best_cost:.6f}")

# ---- Animation ----
fig, ax1 = plt.subplots(figsize=(8,6))
plt.subplots_adjust(bottom=0.25)

# plot placeholders
current_line, = ax1.plot([], [], '-o', color="blue", label="Current")
best_line, = ax1.plot([], [], '-o', color="green", alpha=0.6, label="Best So Far")

# scatter cities
ax1.scatter(city_coords[:, 0], city_coords[:, 1], c="red", s=40)
for idx, (x, y) in enumerate(city_coords):
    ax1.text(x+1, y+1, str(idx), fontsize=8)

metrics_box = ax1.text(0.02, 0.98, "", transform=ax1.transAxes,
                       verticalalignment="top", fontsize=10,
                       bbox=dict(facecolor="white", alpha=0.7))

ax1.legend()

def update(frame_idx):
    tour_indices, cost, gen = history[frame_idx]
    tour_coords = np.vstack([city_coords[tour_indices], city_coords[tour_indices[0]]])
    current_line.set_data(tour_coords[:, 0], tour_coords[:, 1])

    # Best so far
    best_so_far_idx = np.argmin([h[1] for h in history[:frame_idx+1]])
    best_tour_indices = history[best_so_far_idx][0]
    best_tour_coords = np.vstack([city_coords[best_tour_indices], city_coords[best_tour_indices[0]]])
    best_line.set_data(best_tour_coords[:, 0], best_tour_coords[:, 1])

    gap_to_current_best = (cost - best_cost) / best_cost * 100
    metrics_text = (
        f"Generation: {gen}\n"
        f"Current cost: {cost:.2f}\n"
        f"Best cost: {best_cost:.2f}\n"
        f"Gap: {gap_to_current_best:.2f}%"
    )
    metrics_box.set_text(metrics_text)
    return current_line, best_line, metrics_box

ani = FuncAnimation(fig, update, frames=len(history), interval=200,
                    blit=False, repeat=False)

# ---- Controls ----
ax_pause = plt.axes([0.4, 0.05, 0.15, 0.05])
ax_speed = plt.axes([0.65, 0.05, 0.25, 0.03])
pause_button = Button(ax_pause, 'Pause/Resume')
speed_slider = Slider(ax_speed, 'Speed', 0.25, 5.0, valinit=1.0, valstep=0.25)

is_paused = False

def toggle_pause(event):
    if is_paused:
        ani.event_source.start()
    else:
        ani.event_source.stop()
    is_paused = not is_paused

def update_speed(val):
    new_interval = int(200 / val)
    ani.event_source.interval = new_interval

pause_button.on_clicked(toggle_pause)
speed_slider.on_changed(update_speed)

plt.show()
