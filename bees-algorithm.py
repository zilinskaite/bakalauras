import random
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import os

start_time = time.time()

num_dots = 1000
max_coord = 150
a_max = 30
b_max = 10
theta_max = math.pi
num_bees = 100
num_scouts = 30  # Number of scout bees to introduce
num_onlookers = 5  # Number of onlooker bees to introduce
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


def generate_scouts(num_scouts, max_coord, a_max, b_max, theta_max):
    scouts = [(random.uniform(0, max_coord),
               random.uniform(0, max_coord),
               random.uniform(0, a_max),
               random.uniform(0, b_max),
               random.uniform(0, theta_max))
              for _ in range(num_scouts)]
    return scouts


def generate_onlookers(num_onlookers, max_coord, a_max, b_max, theta_max):
    onlookers = [(random.uniform(0, max_coord),
                  random.uniform(0, max_coord),
                  random.uniform(0, a_max),
                  random.uniform(0, b_max),
                  random.uniform(0, theta_max))
                 for _ in range(num_onlookers)]
    return onlookers


def ellipse(x, y, cx, cy, a, b, theta):
    t = np.linspace(0, 2 * np.pi, 1000)
    x_ellipse = cx + a * np.cos(t) * np.cos(theta) - b * np.sin(t) * np.sin(theta)
    y_ellipse = cy + a * np.cos(t) * np.sin(theta) + b * np.sin(t) * np.cos(theta)
    return any(x == x_ and y == y_ for x_, y_ in zip(x_ellipse, y_ellipse))


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


def artificial_bee_algorithm(file_path, a_max, b_max, theta_max, num_bees, num_scouts, num_onlookers, tolerance, max_iterations):
    with open(file_path, 'r') as f:
        points = []
        for y, line in enumerate(f.readlines()):
            points.extend((x, y) for x, val in enumerate(line.strip().split()) if val == '1')
            height = y + 1
            dots = np.array(points)

    def distance_sum(bee):
        cx, cy, a, b, theta = bee
        x_diff = dots[:, 0] - cx
        y_diff = dots[:, 1] - cy
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        x1 = x_diff * cos_theta + y_diff * sin_theta
        y1 = -(x_diff * sin_theta) + y_diff * cos_theta
        x1 /= a
        y1 /= b
        distances = np.sqrt((np.abs(x1) ** 2) + (np.abs(y1) ** 2)) - 1
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

        scouts = generate_scouts(num_scouts, max_coord, a_max, b_max, theta_max)
        scout_fitness = [distance_sum(bee) for bee in scouts]
        best_scout_idx = np.argmin(scout_fitness)
        best_scout_fitness = scout_fitness[best_scout_idx]
        
        if best_scout_fitness < best_fitness:
            best_bee = scouts[best_scout_idx]
            best_fitness = best_scout_fitness
            best_iteration = num_iterations

        if best_fitness < tolerance:
            break

        onlookers = generate_onlookers(num_onlookers, max_coord, a_max, b_max, theta_max)
        onlooker_fitness = [distance_sum(bee) for bee in onlookers]
        for j in range(num_bees):
            best_onlooker_idx = np.argmin(onlooker_fitness)
            best_onlooker_fitness = onlooker_fitness[best_onlooker_idx]
            if best_onlooker_fitness < fitness[j]:
                bees[j] = onlookers[best_onlooker_idx]
                fitness[j] = best_onlooker_fitness

    return best_bee, best_fitness, best_iteration

results = []

def run_multiple_files(input_folder, output_file, num_runs):
    best_fitness_values = [] 
    labels = [] 
    all_fitness_values = []
    all_time_values = []
    
    with open(output_file, 'w') as out_f:
        input_files = os.listdir(input_folder)
        input_files = [file for file in input_files if file.endswith('.txt')]
        input_files = sorted(input_files)[:100]  # Process the first 100 files
        
        for file_name in input_files:
            file_path = os.path.join(input_folder, file_name)
            print(f"Processing {file_name}")
            file_fitness_values = []
            file_time_values = []
            
            for run in range(num_runs):
                start_time = time.time()
                best_bee, best_fitness, best_iteration = artificial_bee_algorithm(file_path, a_max, b_max, theta_max, num_bees, num_scouts, num_onlookers, tolerance, max_iterations)
                elapsed_time = time.time() - start_time
                
                file_fitness_values.append(best_fitness)
                file_time_values.append(elapsed_time)
                all_fitness_values.append(best_fitness)
                all_time_values.append(elapsed_time)
            
            avg_fitness = sum(file_fitness_values) / num_runs
            avg_time = sum(file_time_values) / num_runs
            
            result = {
                'File': file_name,
                'Time': avg_time,
                'Fitness': avg_fitness,
                'BestIteration': best_iteration
            }
            results.append(result)
            
            out_f.write(f"File: {file_name}, Average Fitness: {avg_fitness}, Average Time: {avg_time}\n")
            
            best_fitness_values.append(avg_fitness)
            label = int(file_name.split("_")[-1].split(".")[0])
            labels.append(label)
    
    plt.figure(figsize=(12, 6))
    plt.bar(labels, best_fitness_values)
    plt.xlabel('Data Files')
    plt.ylabel('Best Fitness (Average)')
    plt.title('Average Best Fitness in Each File')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('fitness_plot.png')
    plt.show()
    
    avg_fitness_all = sum(all_fitness_values) / len(all_fitness_values)
    avg_time_all = sum(all_time_values) / len(all_time_values)
    print("Average Fitness across all files and runs:", avg_fitness_all)
    print("Average Time across all files and runs:", avg_time_all)

input_folder = r'C:\Users\zilin\Desktop\Bakalauras\data\1010'
output_file = r'C:\Users\zilin\Desktop\Bakalauras\data\results\abc1010.txt'
num_runs = 1

run_multiple_files(input_folder, output_file, num_runs)
