import numpy as np
import pygad
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider

# ---- Setup problem ----
num_cities = 20
np.random.seed(42)
city_coords = np.random.rand(num_cities, 2) * 100  # random cities
dist_matrix = np.sqrt(((city_coords[:, None, :] - city_coords[None, :, :])**2).sum(axis=2))

def fitness_func(ga, solution, sol_idx):
    tour_length = sum(
        dist_matrix[int(solution[i]), int(solution[(i+1) % len(solution)])]
        for i in range(len(solution))
    )
    return -tour_length

gene_space = list(range(num_cities))

# store history = (tour, cost, generation)
history = []

# ---- Early Stopping State ----
patience = 100
best_so_far = [float("inf")]
no_improvement_counter = [0]

# ---- Callback ----
def callback_generation(ga_instance):
    best_solution, best_fitness, _ = ga_instance.best_solution()
    best_solution = np.array(best_solution, dtype=int)
    best_cost = -best_fitness
    history.append((best_solution, best_cost, ga_instance.generations_completed))

    # check improvement
    if best_cost < best_so_far[0]:
        best_so_far[0] = best_cost
        no_improvement_counter[0] = 0
    else:
        no_improvement_counter[0] += 1

    # early stopping
    if no_improvement_counter[0] >= patience:
        print(f"Stopped early at generation {ga_instance.generations_completed}")
        print("Best solution fitness:", best_fitness)
        print("Best path:", best_solution)

        # Trick for older PyGAD: set generations_completed = num_generations
        ga_instance.generations_completed = ga_instance.num_generations


# ---- GA ----
ga = pygad.GA(
    num_generations=2000,
    num_parents_mating=20,
    fitness_func=fitness_func,
    sol_per_pop=250,
    num_genes=num_cities,
    gene_space=gene_space,
    parent_selection_type="rank",
    keep_parents=5,
    crossover_type="scattered",
    mutation_type="swap",
    mutation_percent_genes=20,
    allow_duplicate_genes=False,
    on_generation=callback_generation
)

# ---- Run ----
start = time.time()
ga.run()
elapsed = time.time() - start

best_solution, best_fitness, _ = ga.best_solution()
best_cost = -best_fitness
print(f"Time Passed: {elapsed:.6f} seconds")
print(f"Best Tour Cost: {best_cost:.6f}")
print("Best Path:", best_solution)

# store full paths history
paths = np.array([h[0] for h in history])
print("Shape of paths array:", paths.shape)

# ---- Animation ----
fig, ax1 = plt.subplots(figsize=(8, 6))
plt.subplots_adjust(bottom=0.25)

# plot placeholders
current_line, = ax1.plot([], [], '-o', color="blue", label="Current")
best_line, = ax1.plot([], [], '-o', color="green", alpha=0.6, label="Best So Far")

# scatter cities
ax1.scatter(city_coords[:, 0], city_coords[:, 1], c="red", s=40)
for idx, (x, y) in enumerate(city_coords):
    ax1.text(x+1, y+1, str(idx), fontsize=8)

metrics_box = ax1.text(
    0.02, 0.98, "", transform=ax1.transAxes,
    verticalalignment="top", fontsize=10,
    bbox=dict(facecolor="white", alpha=0.7)
)

ax1.legend()

# Precompute best-so-far indices for efficiency
best_indices = []
current_best = float("inf")
for i, (_, cost, _) in enumerate(history):
    if cost < current_best:
        current_best = cost
        best_indices.append(i)
    else:
        best_indices.append(best_indices[-1])

def update(frame_idx):
    tour_indices, cost, gen = history[frame_idx]
    tour_coords = np.vstack([city_coords[tour_indices], city_coords[tour_indices[0]]])
    current_line.set_data(tour_coords[:, 0], tour_coords[:, 1])

    # Best so far at this frame
    best_tour_indices = history[best_indices[frame_idx]][0]
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

paused = {"value": False}
def toggle_pause(event):
    if paused["value"]:
        ani.event_source.start()
    else:
        ani.event_source.stop()
    paused["value"] = not paused["value"]

def update_speed(val):
    ani.event_source.interval = int(200 / val)

pause_button.on_clicked(toggle_pause)
speed_slider.on_changed(update_speed)

plt.show()
