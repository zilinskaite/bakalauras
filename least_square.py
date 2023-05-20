import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os
import time
import pandas as pd

def read_points_from_file(file_path):
    points = []
    with open(file_path, 'r') as f:
        content = f.readlines()
        height = len(content)
        for y, line in enumerate(content):
            row = line.strip().split()
            if y == 0:
                width = len(row)
            for x, val in enumerate(row):
                if val == '1':
                    points.append((x, y))
    return np.array(points), width, height

def fit_ellipse_to_points(points, width, height):
    # Define the ellipse equation
    def ellipse_equation(params, x, y):
        a, b, x0, y0, angle = params
        ct = np.cos(angle)
        st = np.sin(angle)
        x_rot = (x - x0) * ct + (y - y0) * st
        y_rot = (y - x0) * st + (y - y0) * ct
        return (x_rot / a) ** 2 + (y_rot / b) ** 2 - 1

    # Define the objective function for least squares
    def objective_function(params, x, y):
        return np.sum(ellipse_equation(params, x, y) ** 2)

    # Improved initial guess for ellipse parameters
    mean_x, mean_y = points.mean(axis=0)
    cov_matrix = np.cov(points.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    a_init, b_init = np.sqrt(5.991 * eigenvalues)
    angle_init = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])

    init_params = [a_init, b_init, mean_x, mean_y, angle_init]

    # Constraints to ensure the fitted ellipse stays within the plane
    def constraint_fun(params):
        a, b, x0, y0, angle = params
        ct = np.cos(angle)
        st = np.sin(angle)
        max_x = max(a * ct, b * st)
        max_y = max(a * st, b * ct)
        return [
            x0 - max_x,
            y0 - max_y,
            width - x0 - max_x,
            height - y0 - max_y,
        ]

    constraints = (
        {'type': 'ineq', 'fun': lambda params: constraint_fun(params)[0]},
        {'type': 'ineq', 'fun': lambda params: constraint_fun(params)[1]},
        {'type': 'ineq', 'fun': lambda params: constraint_fun(params)[2]},
        {'type': 'ineq', 'fun': lambda params: constraint_fun(params)[3]},
        {'type': 'ineq', 'fun': lambda params: params[0] - 1},
        {'type': 'ineq', 'fun': lambda params: params[1] - 1},
    )

    # Perform least squares fitting
    result = minimize(objective_function, init_params, args=(points[:, 0], points[:, 1]), constraints=constraints)

    # Extract the best fitness found
    best_fitness = result.fun

    return best_fitness

def process_files(input_dir, output_file, num_runs):
    input_files = os.listdir(input_dir)
    input_files = [file for file in input_files if file.endswith('.txt')]
    input_files = sorted(input_files)[:100]  # Process the first 100 files

    elapsed_times = []
    best_fitnesses = []

    results = []

    for file_name in input_files:
        file_path = os.path.join(input_dir, file_name)
        print(f"Processing {file_name}")
        points, width, height = read_points_from_file(file_path)

        file_elapsed_times = []
        file_best_fitnesses = []

        for run in range(num_runs):
            start_time = time.time()
            best_fitness = fit_ellipse_to_points(points, width, height)
            elapsed_time = time.time() - start_time
            file_elapsed_times.append(elapsed_time)
            file_best_fitnesses.append(best_fitness)

        avg_elapsed_time = sum(file_elapsed_times) / num_runs
        avg_best_fitness = sum(file_best_fitnesses) / num_runs

        elapsed_times.extend(file_elapsed_times)
        best_fitnesses.extend(file_best_fitnesses)

        results.append({
            'File': file_name,
            'Time': avg_elapsed_time,
            'Fitness': avg_best_fitness
        })

    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)

    # Calculate average fitness and time
    avg_fitness = sum(best_fitnesses) / len(best_fitnesses)
    avg_time = sum(elapsed_times) / len(elapsed_times)
    print("Average Fitness:", avg_fitness)
    print("Average Time Spent:", avg_time)

input_dir = r'C:/Users/zilin/Desktop/Bakalauras/data/1010'
output_file = r'C:/Users/zilin/Desktop/Bakalauras/data/results/linear1010.txt'
num_runs = 1

process_files(input_dir, output_file, num_runs)
