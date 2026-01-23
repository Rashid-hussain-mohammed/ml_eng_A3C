## Deep Reinforcement Learning (A3C) Implementation

# Project Overview (RL)

**Goal**: solving the same optimization problem (Fuel vs. Elasticity) but using an "Artificial Intelligence" approach that learns through trial and error, rather than evolutionary selection.

**Method**: We utilize the A3C (Asynchronous Advantage Actor-Critic) algorithm. Instead of evolving a population, we train a Neural Network to predict the best design variables.

# How our Algorithm works (Basic Explanation)

Think of this algorithm as a team of workers reporting to a manager (Global Brain), rather than a single person trying to solve the problem alone.

    The Global Network (The Master Brain): We start with a central Neural Network that holds the "knowledge" of how to design the car.

    The Workers (Parallel Agents): We create multiple "Worker" agents (e.g., 4 or 8) that run at the same time. Each worker has its own copy of the car simulation.

    Independent Exploration: Each worker takes the current knowledge from the Master Brain and tries out a design. They observe the result (Fuel and Elasticity) and calculate a Reward.

        High Reward: Low Fuel, High Elasticity.

        Low Reward: High Fuel, Low Elasticity.

    **Asynchronous Update**: When a worker finishes a simulation, it calculates what went right or wrong and sends a "Learning Update" (Gradient) to the Master Brain. The Master Brain updates its weights immediately.

    **Syncing**: The worker then downloads the new, smarter weights from the Master Brain and tries again. Because many workers are exploring different designs at the same time, the AI learns much faster and doesn't get stuck in bad designs easily.

**The "Actor" and the "Critic"**:

    The Actor part of the brain decides which gear ratios to pick.

    The Critic part estimates how good that decision was, helping the Actor improve over time.

    Engineering Decisions & Constraints

**To make the AI work with the specific ConsumptionCar.exe, we implemented the following logic**:

    Inputs (State): The current design configuration (normalized).

    Outputs (Action): 5 values corresponding to Final Drive, Gear 1, Gear 3, Gear 4, Gear 5.

    Hard Constraint: The simulation crashes if gears are not strictly ordered (ig3​>ig4​>ig5​).

        Solution: We implemented a wrapper that automatically sorts the gear outputs from the Neural Network before sending them to the simulation.

    **Reward Function**: Reward = (Sum of Elasticities) - (Fuel_Consumption * 2.0)
    We penalize Fuel heavily to ensure the car is efficient.

    Running the A3C Code

To run the Reinforcement Learning agent:

    Ensure ConsumptionCar.exe is in the src/ folder.

    Run the training script: python src/run_a3c.py

**Visualizing Results**: The script will generate optimization_results_A3C.png showing the learning curve.

    Phase 1: Random guessing (low reward).

    Phase 2: Learning (upward trend).

    Phase 3: Convergence (stable high reward).

## Repository Structure
This project follows a standard Python engineering structure.

```text
├── PHASES.md               # Detailed Design Log
├── README.md               # Overview
├── requirements.txt        # Dependencies (tensorflow, numpy, matplotlib)
├── results/                
│   ├── pareto_front.png    # NSGA-II Results
│   └── optimization_results_A3C.png # A3C Learning Curve
└── src/                    
    ├── ConsumptionCar.exe  # The Simulation File
    ├── model.py            # Physics Model wrapper
    ├── run_nsga2.py        # Evolutionary Algorithm (Deliverable 3)
    └── run_a3c.py          # [NEW] A3C Algorithm (Deliverable 2)```