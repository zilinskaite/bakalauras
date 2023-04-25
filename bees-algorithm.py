import random
import math
import numpy as np
import matplotlib.pyplot as plt

# Set up the problem
width = 64
height = 64
num_dots = 100 # Number of randomly generated dots
max_coord = 64 # Maximum coordinate value for dots
a_max = 32 # Maximum value for semi-major axis a
b_max = 32 # Maximum value for semi-minor axis b
theta_max = math.pi # Maximum value for angle of rotation theta

# Generate random dots
dots = [(random.uniform(0, max_coord), random.uniform(0, max_coord)) for i in range(num_dots)]

# Define the ellipse
def ellipse(x, y, a, b, theta):
    # Check if the point is within the plot frame
    if (x < 0).any() or (x >= width).any() or (y < 0).any() or (y >= height).any():
        return 1e6
    x1 = x * math.cos(theta) + y * math.sin(theta)
    y1 = -x * math.sin(theta) + y * math.cos(theta)
    return x1**2 / a**2 + y1**2 / b**2

# Calculate the sum of distances between the ellipse and the dots
def distance_sum(a, b, theta):
    sum_dist = 0
    for dot in dots:
        x, y = dot
        d = math.sqrt(ellipse(x, y, a, b, theta))
        sum_dist += d
    return sum_dist

# Artificial bee algorithm
num_bees = 30 # Number of bees in the population
num_iterations = 100 # Number of iterations

# Initialize the population of bees with random solutions
bees = []
for i in range(num_bees):
    a = random.uniform(5, a_max)
    b = random.uniform(5, b_max)
    theta = random.uniform(0, theta_max)
    bees.append((a, b, theta))

# Evaluate the quality of each solution
fitness = []
for bee in bees:
    a, b, theta = bee
    fitness.append(distance_sum(a, b, theta))

# Main loop
for i in range(num_iterations):
    # Select the best solution
    best_bee = bees[np.argmin(fitness)]

# Generate new solutions around the best solution

new_bees = []
for j in range(num_bees):
    a = best_bee[0] + random.uniform(-1, 1) * a_max / 10
    b = best_bee[1] + random.uniform(-1, 1) * b_max / 10
    theta = best_bee[2] + random.uniform(-1, 1) * theta_max / 10
    new_bees.append((a, b, theta))

# Evaluate the quality of the new solutions
new_fitness = []
for bee in new_bees:
    a, b, theta = bee
    new_fitness.append(distance_sum(a, b, theta))

# Replace the worst solutions in the population
for j in range(num_bees):
    if new_fitness[j] < fitness[j]:
        bees[j] = new_bees[j]
        fitness[j] = new_fitness[j]

# Print the best solution so far
best_fitness = min(fitness)
best_bee = bees[np.argmin(fitness)]
print("Iteration", i, "Best solution:", best_bee, "Fitness:", best_fitness)

# Plot the best ellipse and the dots
a, b, theta = best_bee
x = np.linspace(0, 64, 100)
y = np.linspace(0, 64, 100)
X, Y = np.meshgrid(x, y)
Z = ellipse(X, Y, a, b, theta)
fig, ax = plt.subplots(figsize=(4, 4))
ax.set_xlim(0, 64)
ax.set_ylim(0, 64)
ax.contour(X, Y, Z, levels=[1])
ax.scatter([dot[0] for dot in dots], [dot[1] for dot in dots], s=5)
plt.show()

