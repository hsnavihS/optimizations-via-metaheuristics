import random
import datetime
import os
import matplotlib.pyplot as plt
from fitness_function import fitness_function


def hill_climbing(fitness_func, step_size, max_iterations):
    '''
    Implementation of the hill climbing algorithm to find the maximum confidence that can be achieved from the given facial recognition algorithm by varying scale factor.

    @param fitness_func: the fitness function to be used to evaluate the fitness of each individual
    @param step_size: the size of the step to be taken in the direction of the gradient
    @param max_iterations: the maximum number of iterations to be performed
    '''

    # choose a random initial parameter value within the range of 1 to 2
    current_param = random.uniform(1, 2)
    current_fitness = fitness_func(current_param)

    print(
        f'Initial scale factor: {current_param:.4f}, initial confidence: {current_fitness:.4f}\n')

    # Lists to store the scale factor and confidence values for plotting
    scale_factors = [current_param]
    confidences = [current_fitness]

    now = datetime.datetime.now()
    # Format the date and time as "DD-MM-YYYY at HH:MM"
    formatted_datetime = now.strftime("%d-%m-%Y at %H-%M-%S")

    # create a directory for saving the plots
    plots_dir = f'Hill Climb Plots [{formatted_datetime}]'
    os.makedirs(plots_dir, exist_ok=True)

    # repeat until maximum number of iterations is reached or no improvement is made
    for i in range(max_iterations):
        # choose a random step in the range of -step_size to +step_size
        step = random.uniform(-step_size, step_size)

        dict = {}

        # evaluate the fitness of the new parameter value
        new_param_add = current_param + step
        new_param_sub = current_param - step

        new_fitness_add = fitness_func(new_param_add)
        dict[new_fitness_add] = new_param_add
        new_fitness_sub = fitness_func(new_param_sub)
        dict[new_fitness_sub] = new_param_sub

        new_fitness = max(new_fitness_add, new_fitness_sub)

        # if the new parameter value is better, update the current parameter
        if new_fitness > current_fitness:
            current_param = dict[new_fitness]
            current_fitness = new_fitness

        # Append the current scale factor and confidence values to the lists
        scale_factors.append(current_param)
        confidences.append(current_fitness)

        print(
            f'Iteration: {i + 1}, current scale factor: {current_param:.4f}, current confidence: {current_fitness:.4f}')

    # plot the scale factor and confidence values

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

    # Plot the scale factor on the first subplot
    ax1.plot(scale_factors, label='Scale Factor')
    ax1.set_title('Hill Climbing Optimization')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Scale Factor')

    # Plot the confidence on the second subplot
    ax2.plot(confidences, label='Confidence')
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Confidence')

    plt.subplots_adjust(hspace=0.4)
    plt.savefig(f'{plots_dir}/Scale Factor vs Confidence.png')

    # return the best parameter value found
    return current_param, current_fitness


param, fitness = hill_climbing(fitness_function, 0.1, 40)
print(f"\nBest scale factor: {param:.4f}")
print(f"Maximum confidence: {fitness:.4f}")
