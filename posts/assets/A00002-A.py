import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from common import llm_unstructured_query
import uuid
import os
import matplotlib.animation as animation

MODELS = ["ollama/llama3.2:1b", "ollama/llama3.2:3b","ollama/llama3.1:8b", "ollama/gemma2:2b", "ollama/gemma2:9b"]
TEMP = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
COLORS = ['blue', 'green', 'red', 'purple', 'orange']  # Colors for different runs

# class RandomWalkNN(nn.Module):
#     def __init__(self, input_size=3, hidden_size=16, output_size=2):
#         super(RandomWalkNN, self).__init__()
#         self.network = nn.Sequential(
#             nn.Linear(input_size, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, output_size),
#             nn.Tanh()
#         )

#     def forward(self, x):
#         return self.network(x)

def random_walk_step_llm(t, current_position, model_name, temperature, grid_size):
    """Updated to include grid size information in the prompt"""
    answer = llm_unstructured_query(
        f"You are a random walker in a {grid_size}x{grid_size} grid centered at (0,0). "
        f"At t=0 you started at the center (0,0). Currently at t={t}, your position is {current_position}. "
        f"Reply with either UP, DOWN, LEFT, or RIGHT to move in that direction. "
        f"If you could not comply to prompt, you will stay at the same place.",
        model=model_name,
        temperature=temperature
    )

    if "UP" in answer:
        dx, dy = 0, 1
    elif "DOWN" in answer:
        dx, dy = 0, -1
    elif "LEFT" in answer:
        dx, dy = -1, 0
    elif "RIGHT" in answer:
        dx, dy = 1, 0
    else:
        dx, dy = 0, 0

    return dx, dy

def simulate_random_walk(steps, grid_size=100, start=(0, 0), model_name=None, temperature=0.6):
    path = [start]
    current_position = start

    for t in range(steps):
        dx, dy = random_walk_step_llm(t, current_position, model_name, temperature, grid_size)

        next_position = (current_position[0] + dx, current_position[1] + dy)

        # Ensure the walk stays within the grid
        next_position = (
            max(-grid_size, min(grid_size, next_position[0])),
            max(-grid_size, min(grid_size, next_position[1]))
        )

        current_position = next_position
        path.append(current_position)

    return path

def create_animation(paths, grid_size, output_filename, model_name, temperature, steps):
    """
    Creates an animation of the random walks.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(-grid_size, grid_size)
    ax.set_ylim(-grid_size, grid_size)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', alpha=0.3)

    # Initialize empty line objects for each path
    lines = [ax.plot([], [], color=COLORS[i], lw=2, alpha=0.7, label=f"Run {i+1}")[0] 
            for i in range(len(paths))]
    points = [ax.plot([], [], 'ko', markersize=4, alpha=0.5)[0] 
             for _ in range(len(paths))]
    
    # Plot start point (same for all paths)
    ax.plot(0, 0, 'go', markersize=3, label="Start")

    plt.title(f"Random Walk Animation\nModel: {model_name}\nTemperature: {temperature}, Steps: {steps}", pad=20)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    def init():
        for line, point in zip(lines, points):
            line.set_data([], [])
            point.set_data([], [])
        return lines + points

    def animate(frame):
        for i, (path, line, point) in enumerate(zip(paths, lines, points)):
            if frame < len(path):
                x_vals, y_vals = zip(*path[:frame+1])
                line.set_data(x_vals, y_vals)
                point.set_data([x_vals[-1]], [y_vals[-1]])
            else:
                x_vals, y_vals = zip(*path)
                line.set_data(x_vals, y_vals)
                point.set_data([x_vals[-1]], [y_vals[-1]])
        return lines + points

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                 frames=steps + 10,  # Extra frames to show final state
                                 interval=500,  # 500ms between frames
                                 blit=True)
    
    # Save animation
    anim.save(output_filename, writer='ffmpeg', fps=2, dpi=300, 
              extra_args=['-vcodec', 'libx264'])
    plt.close(fig)

def save_random_walk_image(paths, grid_size=100, filename="random_walk.png", model_name="", temperature=0.0, steps=10):
    """
    Saves multiple random walk paths on the same plot.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(-grid_size, grid_size)
    ax.set_ylim(-grid_size, grid_size)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', alpha=0.3)

    for i, path in enumerate(paths):
        x_vals, y_vals = zip(*path)
        ax.plot(x_vals, y_vals, color=COLORS[i], lw=2, alpha=0.7, label=f"Run {i+1}")
        ax.scatter(x_vals[1:-1], y_vals[1:-1], color='black', s=20, alpha=0.5)
        ax.plot(x_vals[0], y_vals[0], 'go', markersize=10, label="Start" if i == 0 else "")
        ax.plot(x_vals[-1], y_vals[-1], 'ro', markersize=10, label="End" if i == 0 else "")

    plt.title(f"Random Walk\nModel: {model_name}\nTemperature: {temperature}, Steps: {steps}", pad=20)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    steps = 50
    grid_size = 25
    num_runs = 5
    
    uid = str(uuid.uuid4())
    print(f"UID: {uid}")
    
    # Create directory structure
    base_dir = f"data/{uid}"
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(f"{base_dir}/images", exist_ok=True)
    os.makedirs(f"{base_dir}/animations", exist_ok=True)

    for model_name in MODELS:
        for temperature in TEMP:
            print(f"Simulating for model={model_name}, temperature={temperature}")
            
            paths = []
            for run in range(num_runs):
                print(f"  Run {run + 1}/{num_runs}")
                path = simulate_random_walk(steps, grid_size=grid_size, 
                                         model_name=model_name, temperature=temperature)
                paths.append(path)
            
            # Save static image
            image_filename = f"{base_dir}/images/random_walk_{model_name.replace('/', '_')}_temp_{temperature}.png"
            save_random_walk_image(paths, grid_size=grid_size, filename=image_filename,
                                 model_name=model_name, temperature=temperature, steps=steps)
            
            # Create and save animation
            animation_filename = f"{base_dir}/animations/random_walk_{model_name.replace('/', '_')}_temp_{temperature}.mp4"
            create_animation(paths, grid_size, animation_filename, model_name, temperature, steps)
