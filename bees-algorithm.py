import random
import math
import numpy as np
import matplotlib.pyplot as plt
import time
import os

start_time = time.time()

num_dots = 1000
max_coord = 64
a_max = 30
b_max = 10
theta_max = math.pi
num_bees = 20
tolerance = 1e-6
max_iterations = 5000

def initialize_population(num_bees, a_max, b_max, theta_max):
    bees = [(random.uniform(0, max_coord),
             random.uniform(0, max_coord),
             random.uniform(0, a_max),
             random.uniform(0, b_max),
             random.uniform(0, theta_max))
            for _ in range(num_bees)]
    return bees

def ellipse(x, y, cx, cy, a, b, theta):
    t = np.linspace(0, 2*np.pi, 1000)
    x_ellipse = cx + a*np.cos(t)*np.cos(theta) - b*np.sin(t)*np.sin(theta)
    y_ellipse = cy + a*np.cos(t)*np.sin(theta) + b*np.sin(t)*np.cos(theta)
    return (x, y) in zip(x_ellipse, y_ellipse)

def update_bees(bees, best_bee, num_bees, max_coord, a_max, b_max, theta_max, step_scale):
    new_bees = []
    for _ in range(num_bees):
        new_bee = (best_bee[0] + random.uniform(-1, 1) * max_coord * step_scale,
                   best_bee[1] + random.uniform(-1, 1) * max_coord * step_scale,
                   best_bee[2] + random.uniform(-1, 1) * a_max * step_scale,
                   best_bee[3] + random.uniform(-1, 1) * b_max * step_scale,
                   best_bee[4] + random.uniform(-1, 1) * theta_max * step_scale)
        new_bees.append(new_bee)
    return new_bees

def artificial_bee_algorithm(file_path, a_max, b_max, theta_max, num_bees, tolerance, max_iterations):
    with open(file_path, 'r') as f:
        points = []
        for y, line in enumerate(f.readlines()):
            for x, val in enumerate(line.strip().split()):
                if val == '1':
                    points.append((x, y))

    height = y + 1  # Calculate the height based on the number of rows
    dots = np.array(points)

    def distance_sum(bee):
        cx, cy, a, b, theta = bee
        distances = []
        for x, y in dots:
            x1 = (x - cx) * np.cos(theta) + (y - cy) * np.sin(theta)
            y1 = -(x - cx) * np.sin(theta) + (y - cy) * np.cos(theta)
            x1, y1 = x1 / a, y1 / b
            distance = np.sqrt((x1**2) + (y1**2)) - 1
            distances.append(distance)
        return np.sum(np.abs(distances))

    bees = initialize_population(num_bees, a_max, b_max, theta_max)
    best_fitness = float('inf')
    num_iterations = 0
    best_iteration = 0
    fitness_values = []

    while num_iterations < max_iterations:
        num_iterations += 1
        fitness = [distance_sum(bee) for bee in bees]
        best_bee_idx = np.argmin(fitness)
        best_bee = bees[best_bee_idx]
        best_fitness = fitness[best_bee_idx]
        fitness_values.append(best_fitness)

        if num_iterations > 1 and abs((best_fitness - prev_best_fitness) / prev_best_fitness) < tolerance:
            break

        prev_best_fitness = best_fitness
        step_scale = 1 / (num_iterations + 1)
        new_bees = update_bees(bees, best_bee, num_bees, max_coord, a_max, b_max, theta_max, step_scale)

        new_fitness = [distance_sum(bee) for bee in new_bees]
        for j in range(num_bees):
            if new_fitness[j] < fitness[j]:
                bees[j] = new_bees[j]
                fitness[j] = new_fitness[j]
                best_iteration = num_iterations
        print("Iteration", num_iterations, "Best solution:", best_bee, "Fitness:", best_fitness)
        if best_fitness < tolerance:
            break
    print("--- %s seconds ---" % (time.time() - start_time))

    #cx, cy, a, b, theta = best_bee
    #t = np.linspace(0, 2 * np.pi, 1000)
    #x_ellipse = cx + a * np.cos(t) * np.cos(theta) - b * np.sin(t) * np.sin(theta)
    #y_ellipse = cy + a * np.cos(t) * np.sin(theta) + b * np.sin(t) * np.cos(theta)
    #fig, ax = plt.subplots()
    #ax.plot(x_ellipse, height - y_ellipse - 1, 'r')  # Flip the y-coordinates and subtract 1 to align with the coordinate system
    #ax.scatter(dots[:, 0], height - dots[:, 1] - 1, s=5)  # Flip the y-coordinates and subtract 1
    #plt.gca().set_aspect('equal', adjustable='box')
    #plt.show()

    #plt.plot(range(1, len(fitness_values) + 1), fitness_values)
    #plt.xlabel('Iteration')
    #plt.ylabel('Fitness Value')
    #plt.title('Convergence Analysis')
    #plt.show()

    return best_bee, best_fitness, best_iteration

def run_multiple_files(input_folder, output_file):
    with open(output_file, 'w') as out_f:
        input_files = os.listdir(input_folder)
        input_files = [file for file in input_files if file.endswith('.txt')]
        input_files = sorted(input_files)[:100]  # Process the first 100 files

        for file_name in input_files:
            file_path = os.path.join(input_folder, file_name)
            print(f"Processing {file_name}")
            best_bee, best_fitness, best_iteration = artificial_bee_algorithm(file_path, a_max, b_max, theta_max, num_bees, tolerance, max_iterations)
            out_f.write(f"File: {file_name}, Best solution: {best_bee}, Fitness: {best_fitness}, Best iteration: {best_iteration}\n")



input_folder = r'C:\Users\zilin\Desktop\Bakalauras\data\210'
output_file = r'C:\Users\zilin\Desktop\Bakalauras\results\abc210.txt'

run_multiple_files(input_folder, output_file)

#print("Best solution:", best_bee, "Fitness:", best_fitness, "Best iteration:", best_iteration)


