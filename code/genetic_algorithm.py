import os
import random
import datetime
import matplotlib.pyplot as plt
from fitness_function import fitness_function


def genetic_algorithm(population_size, fitness_func, num_generations, mutation_rate, mode):
    '''
    Implementation of a genetic algorithm to find the fittest individual in a population
    to be used for parameter tuning of a machine learning model for facial recognition.

    @param population_size: the number of individuals in the population
    @param fitness_func: the fitness function to be used to evaluate the fitness of each individual
    @param num_generations: the number of generations to evolve the population
    @param mutation_rate: the probability of mutation of an offspring
    @param mode: the mode of the genetic algorithm to be used (1: show plot after every generation
    2: save a plot showing the param values vs fitness scores for each individual in every generation)
    '''

    # create an initial population of random parameter values
    population = [random.uniform(1, 2) for _ in range(population_size)]
    print(f'Initial population: {population}')
    initial_fitness_scores = [fitness_func(param) for param in population]
    print(f'Initial fitness scores: {initial_fitness_scores}\n')
    print('-' * 180, '\n')

    now = datetime.datetime.now()
    # Format the date and time as "DD-MM-YYYY at HH:MM"
    formatted_datetime = now.strftime("%d-%m-%Y at %H-%M-%S")

    if (mode == 2):
        # create a directory for saving the plots
        plots_dir = f'GA Plots [{formatted_datetime}]'
        os.makedirs(plots_dir, exist_ok=True)

    generation_number = 0
    for i in range(num_generations):
        print(f'Generation #{i + 1}:\n')
        # evaluate fitness of each individual in the population
        fitness_scores = [fitness_func(param) for param in population]
        print(f'Population:', population)
        print(f'Fitness scores:', fitness_scores, '\n')

        # select the fittest individuals for the next generation
        fittest_indices = sorted(range(population_size), key=lambda i: fitness_scores[i], reverse=True)[
            :int(population_size/2)]
        fittest_population = [population[i] for i in fittest_indices]

        # create the next generation by mating fittest individuals
        next_generation = []
        for i in range(population_size):
            parent1 = random.choice(fittest_population)
            parent2 = random.choice(fittest_population)
            offspring = (parent1 + parent2) / 2.0

            # mutation
            if mutation_rate > random.randint(0, 100) / 100:
                offspring = random.uniform(1, 2)

            next_generation.append(offspring)

        population = next_generation
        print(f'Offsprings:', population, '\n')
        print('-' * 180, '\n')

        # plot param values vs fitness scores for each individual in the current generation
        plt.scatter(population, fitness_scores)
        plt.xlabel('Param Value')
        plt.ylabel('Fitness Score')
        plt.title(
            f'Scale Factor vs Fitness Scores till generation {generation_number + 1}')
        plt.xlim(1, 2)  # Set x axis limits from 1 to 2
        plt.ylim(0, 100)  # Set y axis limits from 0 to 100
        if mode == 2:
            # save plot as an image
            plt.savefig(
                f'{plots_dir}/Plot till generation {generation_number + 1}.png')
            plt.title('Cumulative Plot')
            plt.savefig(f'{plots_dir}/Cumulative.png')
        else:
            plt.show()

        generation_number += 1

    # return the fittest individual in the final population
    fittest_index = max(range(population_size),
                        key=lambda i: fitness_scores[i])
    fittest_param = population[fittest_index]
    fittest_fitness = fitness_scores[fittest_index]
    return fittest_param, fittest_fitness


def main():
    param, fitness = genetic_algorithm(5, fitness_function, 10, 0.5, 2)
    print("Best value of scale factor: ", param)
    print("Maximum fitness: ", fitness)


if __name__ == '__main__':
    main()
