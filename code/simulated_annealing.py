import os
import random
import datetime
import math
import matplotlib.pyplot as plt
from fitness_function import fitness_function


def simulated_annealing(fitness_func, init_param, init_temp, cool_rate, stopping_temp, max_iterations):
    '''
    Implementation of the simulated annealing algorithm for finding the maximum confidence that can be achieved from the given facial recognition algorithm by varying scale factor.

    @param fitness_func: the fitness function to be used to evaluate the fitness of each individual
    @param init_param: the initial parameter value to be used in the algorithm
    @param init_temp: the initial temperature to be used in the algorithm
    @param cool_rate: the cooling rate to be used in the algorithm
    @param stopping_temp: the stopping temperature to be used in the algorithm
    @param max_iterations: the maximum number of iterations to be performed
    '''

    # initialize the current state with the initial parameter value
    current_param = init_param
    current_fitness, current_prediction = fitness_func(current_param)

    # initialize the best state and its fitness to the current state and its fitness
    best_param = current_param
    best_fitness = current_fitness

    print(
        f'Initial scale factor: {current_param}, initial confidence: {current_fitness}')
    print('\n' + '-' * 180 + '\n')

    # Lists to store the data for plotting
    param_data = [current_param]
    fitness_data = [current_fitness]
    best_param_data = [best_param]
    best_fitness_data = [best_fitness]

    # repeat until stopping temperature or maximum number of iterations is reached
    for i in range(max_iterations):
        print(f'Iteration #{i + 1}:')
        # calculate the current temperature as a function of the cooling rate and the iteration number
        temp = init_temp * math.exp(-cool_rate * i)

        # choose a random new parameter value within the range of 1 to 2 and calculate fitness
        new_param = random.uniform(1, 2)
        new_fitness, new_prediction = fitness_func(new_param)

        # calculate the difference in fitness between the current and new states
        fitness_diff = new_fitness - current_fitness

        # if the new state has higher fitness, accept it as the new current state
        if fitness_diff > 0:
            print(
                f'New state has higher fitness: {new_fitness} > {current_fitness}, accepting new state.')
            current_param = new_param
            current_fitness = new_fitness
            current_prediction = new_prediction
        # otherwise, accept the new state with a probability that depends on the temperature
        else:
            acceptance_prob = math.exp(fitness_diff / temp)
            if new_fitness == 0:
                print('New state has fitness 0, not accepting new state')
            if random.random() < acceptance_prob and new_fitness != 0:
                print(
                    f'New state has lower fitness: {new_fitness} < {current_fitness}, still accepting new state with probability {acceptance_prob}.')
                current_param = new_param
                current_fitness = new_fitness
                current_prediction = new_prediction

        # update the best state if the current state has higher fitness
        if current_fitness > best_fitness:
            best_param = current_param
            best_fitness = current_fitness
            best_prediction = current_prediction

        # Append data for plotting
        param_data.append(current_param)
        fitness_data.append(current_fitness)
        best_param_data.append(best_param)
        best_fitness_data.append(best_fitness)

        print(
            f'Current scale factor: {current_param}, current confidence: {current_fitness}, current prediction: {current_prediction}')
        print(
            f'\nBest scale factor: {best_param}, best confidence: {best_fitness}, best prediction: {best_prediction}\n')

        # stop if the temperature has reached the stopping temperature
        if temp < stopping_temp:
            print(
                f'\nTemperature has reached stopping temperature: {temp} < {stopping_temp}')
            break

        print('-' * 180 + '\n')

    # return the best parameter value found
    return best_param, best_fitness, best_prediction, param_data, fitness_data, best_param_data, best_fitness_data


param, fitness, prediction, param_data, fitness_data, best_param_data, best_fitness_data = simulated_annealing(
    fitness_function, 1.5, 100, 0.01, 0.01, 10)
print("Best parameter: ", param)
print("Best fitness: ", fitness)
print("Prediction: ", prediction)

# Plot the data
now = datetime.datetime.now()
# Format the date and time as "DD-MM-YYYY at HH:MM"
formatted_datetime = now.strftime("%d-%m-%Y at %H-%M-%S")

# create a directory for saving the plots
plots_dir = f'Simulated Annealing Plots [{formatted_datetime}]'
os.makedirs(plots_dir, exist_ok=True)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
ax1.plot(param_data, label='Current Scale Factor')
ax1.plot(best_param_data, label='Best Scale Factor')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Scale Factor')
ax1.legend()

ax2.plot(fitness_data, label='Current Fitness')
ax2.plot(best_fitness_data, label='Best Fitness')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Fitness')
ax2.legend()

plt.tight_layout()
plt.savefig(f'{plots_dir}/Best and current fitness and scale factor.png')
