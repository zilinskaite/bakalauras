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

def ellipse(x, y, cx, cy, a, b, theta):
    t = np.linspace(0, 2 * np.pi, 100)
    ct = np.cos(theta)
    st = np.sin(theta)
    x_rot = (x - cx) * ct + (y - cy) * st
    y_rot = -(x - cx) * st + (y - cy) * ct
    return (x_rot / a) ** 2 + (y_rot / b) ** 2 - 1


def fit_ellipse_to_points(points, width, height, max_ls_iterations):
    # Define the objective function for least squares
    def objective_function(params, points):
        a, b, x0, y0, angle = params
        errors = ellipse(points[:, 0], points[:, 1], x0, y0, a, b, angle)
        return np.sum(errors ** 2)

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
    result = minimize(objective_function, init_params, args=(points,), constraints=constraints,
                      options={'maxiter': max_ls_iterations})

    # Extract the best fitness found
    best_fitness = result.fun

    return result.x, best_fitness


def plot_ellipse(ax, params):
    a, b, x0, y0, angle = params
    t = np.linspace(0, 2 * np.pi, 100)
    ct = np.cos(angle)
    st = np.sin(angle)
    x = x0 + a * np.cos(t) * ct - b * np.sin(t) * st
    y = y0 + a * np.cos(t) * st + b * np.sin(t) * ct
    ax.plot(x, y, color='red')


def process_files(input_dir, output_file, max_ls_iterations):
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

        start_time = time.time()
        params, best_fitness = fit_ellipse_to_points(points, width, height, max_ls_iterations)
        elapsed_time = time.time() - start_time

        elapsed_times.append(elapsed_time)
        best_fitnesses.append(best_fitness)

        results.append({
            'File': file_name,
            'Time': elapsed_time,
            'Fitness': best_fitness
        })

        # Plot the best ellipse found
        fig, ax = plt.subplots()
        ax.scatter(points[:, 0], points[:, 1], color='blue', s=3)
        plot_ellipse(ax, params)
        ax.set_aspect('equal')
        #plt.title(f"Best Ellipse - {file_name}")
        plt.savefig(f"best_ellipse_{file_name}.png")
        plt.show()

    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)

    avg_fitness = sum(best_fitnesses) / len(best_fitnesses)
    avg_time = sum(elapsed_times) / len(elapsed_times)
    print("Average Fitness:", avg_fitness)
    print("Average Time Spent:", avg_time)


input_dir = r'\data\results\test-control'
output_file = r'\data\results\sqr-test.txt'
max_ls_iterations = 5000

process_files(input_dir, output_file, max_ls_iterations)
