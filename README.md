# Optimization using Metaheuristics

Made as part of CS F407: Articial Intelligence; this project explores the application of basic AI optimization algorithms—Genetic Algorithm, Particle Swarm Optimization, Hill Climbing, and Simulated Annealing—to tune parameters in a simple facial recognition task.  
The primary focus is on demonstrating how these algorithms can introduce "intelligence" through randomness, rather than achieving high-accuracy facial recognition.

## Overview

- **GUI**: A minimal Tkinter interface allows users to select an algorithm and input parameters.
- **Backend**: Implements a basic facial recognition system using OpenCV.
- **Objective**: Optimize the "scale factor" parameter to improve the "confidence" metric in facial recognition.
- **Algorithms**: Includes implementations of Genetic Algorithm, Particle Swarm Optimization, Hill Climbing, and Simulated Annealing.

## Setup Instructions

1. **Clone the repository**:

   ```bash
   git clone https://github.com/hsnavihS/AI-Project.git
   cd optimization-via-metaheuristics
   ```

2. Install dependencies and run the GUI:

    ```bash
    pip intall -r requirements.txt
    python ./code/gui.py
    ```
