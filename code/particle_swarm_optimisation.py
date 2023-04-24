import random
import datetime
import os
import matplotlib.pyplot as plt
from fitness_function import fitness_function


def particle_swarm_optimization(fitness_func, num_particles, max_iterations):
    '''
    Implementation of the particle swarm optimization algorithm for finding the maximum confidence that can be extracted from a facial recognition algorithm by varying the scale factor.

    @param fitness_func: the fitness function to be used to evaluate the fitness of each individual
    @param num_particles: the number of particles to be used in the algorithm
    @param max_iterations: the maximum number of iterations to be performed
    '''

    # initialize the particles with random positions within the range of 1 to 2 and random velocities
    particles = []
    print(f'Initial particles:\n')
    for i in range(num_particles):
        position = random.uniform(1, 2)
        velocity = random.uniform(-0.1, 0.1)
        best_fitness, prediction = fitness_func(position)
        particle = {
            "position": position,
            "velocity": velocity,
            "best_position": position,
            "best_fitness": best_fitness,
            "prediction": prediction
        }
        print(
            f'Particle #{i + 1}: position: {particle["position"]:.4f}, velocity: {particle["velocity"]:.4f}, current fitness: {particle["best_fitness"]:.4f}')
        particles.append(particle)
    print('\n' + '-'*150)

    # initialize the global best position and its fitness to the first particle's best position and its fitness
    global_best_position = particles[0]["best_position"]
    global_best_fitness = particles[0]["best_fitness"]
    global_prediction = particles[0]["prediction"]

    # lists to store data for plotting
    global_best_positions = []
    global_best_fitnesses = []
    particle_positions = [[] for _ in range(num_particles)]
    particle_fitnesses = [[] for _ in range(num_particles)]

    now = datetime.datetime.now()
    # Format the date and time as "DD-MM-YYYY at HH:MM"
    formatted_datetime = now.strftime("%d-%m-%Y at %H-%M-%S")

    # create a directory for saving the plots
    plots_dir = f'PSO Plots [{formatted_datetime}]'
    os.makedirs(plots_dir, exist_ok=True)

    # repeat until maximum number of iterations is reached
    for i in range(max_iterations):
        print(f'Iteration #{i + 1}:\n')

        # update the velocity and position of each particle
        for j in range(len(particles)):
            particle = particles[j]

            # update the velocity, limit it to max 0.1
            new_velocity = 0.5 * particle["velocity"] + 1.5 * random.random() * (
                particle["best_position"] - particle["position"]) + 2.0 * random.random() * (global_best_position - particle["position"])
            particle["velocity"] = max(min(new_velocity, 0.1), -0.1)

            # update the position, make sure it stays within the range of 1 to 2
            particle["position"] += particle["velocity"]
            particle["position"] = max(min(particle["position"], 2), 1)

            # update the particle's best position and its fitness if its current position has higher fitness
            current_fitness, current_prediction = fitness_func(
                particle["position"])
            if current_fitness > particle["best_fitness"]:
                particle["best_position"] = particle["position"]
                particle["best_fitness"] = current_fitness
                particle["prediction"] = current_prediction

            # store data for plotting
            particle_positions[j].append(particle["position"])
            particle_fitnesses[j].append(current_fitness)

            print(
                f'Particle #{j + 1}: position: {particle["position"]:.4f}, velocity: {particle["velocity"]:.4f}, current fitness: {current_fitness:.4f}, best_fitness: {particle["best_fitness"]:.4f}, prediction: {particle["prediction"]}')

        # update the global best position and its fitness if any particle has higher fitness than it
        for particle in particles:
            if particle["best_fitness"] > global_best_fitness:
                global_best_position = particle["best_position"]
                global_best_fitness = particle["best_fitness"]
                global_prediction = particle["prediction"]

        global_best_positions.append(global_best_position)
        global_best_fitnesses.append(global_best_fitness)

        print(
            f'Global best position: {global_best_position:.4f}, global best fitness: {global_best_fitness:.4f}, prediction: {global_prediction}\n')
        print('-'*150)

    # plot the global best position and its fitness over iterations
    plt.figure(figsize=(12, 6))
    plt.plot(global_best_positions)
    plt.xlabel('Iteration')
    plt.ylabel('Global Best Position')
    plt.title('Global Best Position over Iterations')
    plt.savefig(f'{plots_dir}/global_best_positions.png')

    plt.figure(figsize=(12, 6))
    plt.plot(global_best_fitnesses)
    plt.xlabel('Iteration')
    plt.ylabel('Global Best Fitness')
    plt.title('Global Best Fitness over Iterations')
    plt.savefig(f'{plots_dir}/global_best_fitnesses.png')

    # plot the position and fitness of all particles in one graph

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    for j in range(num_particles):
        ax1.plot(particle_positions[j], label=f'Particle {j + 1} Position')
        ax2.plot(particle_fitnesses[j], label=f'Particle {j + 1} Fitness')

    # Add labels and titles to the subplots
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Position')
    ax1.set_title('Particle Position over Iterations')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Fitness')
    ax2.set_title('Particle Fitness over Iterations')

    # Add legend to differentiate between position and fitness
    ax1.legend()
    ax2.legend()

    # Save the plot
    plt.savefig(f'{plots_dir}/particle_positions_and_fitnesses.png')

    # plot the position and fitness of each particle in a separate graph

    for j in range(num_particles):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))

        # Plot fitness on the first subplot
        ax1.plot(particle_fitnesses[j])
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Fitness')
        ax1.set_title(f'Particle {j + 1} Fitness over Iterations')

        # Plot position on the second subplot
        ax2.plot(particle_positions[j])
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Position')
        ax2.set_title(f'Particle {j + 1} Position over Iterations')

        # Adjust spacing between subplots
        plt.subplots_adjust(hspace=0.5)

        # Save the plot
        plt.savefig(f'{plots_dir}/particle #{j + 1}.png')

    print('Optimization complete! Plots saved.\n')
    return global_best_position, global_best_fitness, global_prediction


def main():

    # set the number of particles and maximum number of iterations
    num_particles = 5
    max_iterations = 10

    # call the particle_swarm_optimization function with the fitness function and parameters
    param, fitness, prediction = particle_swarm_optimization(
        fitness_function, num_particles, max_iterations)
    print(f"\nBest scale factor: {param:.4f}")
    print(f"Maximum confidence: {fitness:.4f}")
    print(f"Prediction: {prediction}")


if __name__ == '__main__':
    main()
