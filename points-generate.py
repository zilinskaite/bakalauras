import random
import os

# Define the dimensions of the white plane
width = 150
height = 100

# Define the probability of a point being black (0 = always white, 1 = always black)
prob_black = 0.01

# Set the number of files you want to create
num_files = 100

for i in range(1, num_files + 1):
    # Generate the random black points
    points = []
    for x in range(width):
        for y in range(height):
            if random.random() < prob_black:
                points.append((x, y))

    # Encode the points with '0' (white) and '1' (black)
    encoded_points = [['0'] * width for _ in range(height)]
    for x, y in points:
        encoded_points[y][x] = '1'

    # Create a new file and save the encoded points to it
    file_name = f'points_prob1010_{i}.txt'
    file_path = os.path.join(r'C:\Users\zilin\Desktop\Bakalauras\data\1010', file_name)
    
    if os.path.exists(file_path):
        print(f'The file {file_path} already exists.')
    else:
        with open(file_path, 'w') as f:
            for line in encoded_points:
                f.write(' '.join(line) + '\n')
        print(f'The file {file_path} was created and saved successfully.')
