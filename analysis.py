import matplotlib.pyplot as plt
import pandas as pd

def save_results_to_file(results, output_file):
    df = pd.DataFrame(results, columns=['File', 'Fitness (Output 1)', 'Fitness (Output 2)', 'Accuracy'])
    df.to_csv(output_file, index=False)

def analyze_results(output_file1, output_file2):
    # Read the first output file
    with open(output_file1, 'r') as f1:
        lines1 = f1.readlines()[1:]  # Skip the header line
        file_names1, elapsed_times1, best_fitnesses1 = [], [], []
        for line in lines1:
            parts = line.strip().split(',')
            file_name = parts[0].split(':')[1].strip()  # Extract the file name
            best_fitness = float(parts[1].split(':')[1].strip())  # Extract the best fitness value
            file_names1.append(file_name)
            best_fitnesses1.append(best_fitness)

    # Read the second output file
    with open(output_file2, 'r') as f2:
        lines2 = f2.readlines()[1:]  # Skip the header line
        file_names2, elapsed_times2, best_fitnesses2 = [], [], []
        for line in lines2:
            file_name, elapsed_time, best_fitness = line.strip().split(',')
            file_names2.append(file_name)
            elapsed_times2.append(float(elapsed_time))
            best_fitnesses2.append(float(best_fitness))

    accuracies = []
    for fitness1, fitness2 in zip(best_fitnesses1, best_fitnesses2):
        accuracy = (fitness1 * 100) / fitness2
        accuracies.append(accuracy)

    # Calculate the frequency of each accuracy percentage
    accuracy_counts = {}
    for accuracy in accuracies:
        percentage = int(accuracy // 10) * 10
        if percentage in accuracy_counts:
            accuracy_counts[percentage] += 1
        else:
            accuracy_counts[percentage] = 1

    # Generate the x-axis tick labels and accuracy frequencies
    x_ticks = []
    accuracy_frequencies = []
    for i in range(0, 101, 10):
        x_ticks.append(i)
        accuracy_frequencies.append(accuracy_counts.get(i, 0))

    # Create the bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(x_ticks, accuracy_frequencies, tick_label=[f"{x}%" for x in x_ticks])
    plt.xlabel('Accuracy Percentage')
    plt.ylabel('Frequency')
    plt.title('Accuracy Distribution')
    plt.show()

    # Return the results as a list of dictionaries
    results = [
        {
            'File': file_name,
            'Fitness (Output 1)': fitness1,
            'Fitness (Output 2)': fitness2,
            'Accuracy': accuracy
        }
        for file_name, fitness1, fitness2, accuracy in zip(file_names1, best_fitnesses1, best_fitnesses2, accuracies)
    ]
    return results

# Provide the paths of the output files
output_file1 = r'C:\Users\zilin\Desktop\Bakalauras\data\results\abc210-510.txt'
output_file2 = r'C:\Users\zilin\Desktop\Bakalauras\data\results\linear210-510.txt'

results = analyze_results(output_file1, output_file2)

# Save the results to a file
output_file3 = r'C:\Users\zilin\Desktop\Bakalauras\data\results\analysis\210-510.txt'
save_results_to_file(results, output_file3)
