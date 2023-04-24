import tkinter as tk
from tkinter import simpledialog
from hill_climb import hill_climbing
from particle_swarm_optimisation import particle_swarm_optimization
from simulated_annealing import simulated_annealing
from genetic_algorithm import genetic_algorithm
from fitness_function import fitness_function


def on_pso_click():
    num_particles = simpledialog.askinteger(
        "Number of particles", f"Enter Number of particles")
    num_iterations = simpledialog.askinteger(
        "Number of iterations", f"Enter Number of iterations")
    scaleFactor, confidence, label = particle_swarm_optimization(
        fitness_function, num_particles, num_iterations)
    result_label.config(
        text=f"Scale factor = {scaleFactor}, confidence = {confidence}, prediction = {label}")


def on_ga_click():
    population_size = simpledialog.askinteger(
        "Population Size", f"Enter Population size")
    num_generations = simpledialog.askinteger(
        "Number of generations", f"Enter number of generations")
    mutation_rate = simpledialog.askfloat(
        "Mutation rate", f"Enter Mutation rate")
    mode = simpledialog.askinteger("Mode", f"Enter Mode (either 1 or 2)")
    scaleFactor, confidence, label = genetic_algorithm(
        population_size, fitness_function, num_generations, mutation_rate, mode)
    result_label.config(
        text=f"Scale Factor, confidence and label is : {scaleFactor, confidence, label}")


def on_hill_climbing_click():
    step_size = simpledialog.askfloat(
        "Step size", f"Enter step size")
    num_iterations = simpledialog.askinteger(
        "Number of iterations", f"Enter Number of iterations")
    scaleFactor, confidence, label = hill_climbing(
        fitness_function, step_size, num_iterations)
    result_label.config(
        text=f"Scale Factor, confidence and label is : {scaleFactor, confidence, label}")


def on_simulated_annealing_click():
    init_param = simpledialog.askfloat(
        "Initial parameter", f"Enter initial parameter")
    init_temp = simpledialog.askinteger(
        "Initial Temperature", f"Enter Initial Temperature")
    cool_rate = simpledialog.askfloat(
        "Cool rate", f"Enter Cool rate")
    stop_temp = simpledialog.askfloat("Stop temp", f"Enter stop temp")
    num_iterations = simpledialog.askinteger(
        "Number of iterations", f"Enter Number of iterations")
    scaleFactor, confidence, label = simulated_annealing(
        fitness_function, init_param, init_temp, cool_rate, stop_temp, num_iterations)
    result_label.config(
        text=f"Scale Factor, confidence and label is : {scaleFactor, confidence, label}")


def on_close():
    root.quit()


root = tk.Tk()
root.title("Choose search algorithm")

frame = tk.Frame(root)
frame.pack(padx=15, pady=15)

pso_button = tk.Button(frame, text="PSO", command=on_pso_click)
pso_button.grid(row=0, column=0, padx=5, pady=5)

ga_button = tk.Button(frame, text="GA", command=on_ga_click)
ga_button.grid(row=0, column=2, padx=5, pady=5)

hill_climbing_button = tk.Button(
    frame, text="Hill Climbing", command=on_hill_climbing_click)
hill_climbing_button.grid(row=1, column=0, padx=5, pady=5)

simulated_annealing_button = tk.Button(
    frame, text="Simulated Annealing", command=on_simulated_annealing_click)
simulated_annealing_button.grid(row=1, column=2, padx=5, pady=5)

quit_button = tk.Button(
    frame, text="Close", command=on_close)
quit_button.grid(row=2, column=1, padx=5, pady=5)

result_label = tk.Label(frame, text="")
result_label.grid(row=3, column=0, columnspan=2, pady=10)

frame.tkraise()
root.mainloop()
