# Multi-Objective Optimization of Vehicle Design (NSGA-II)

## Project Overview
**Goal:** Optimize passenger vehicle design to balance two conflicting objectives: **minimizing fuel consumption** and **maximizing acceleration performance**.

**Method:** We utilize the **NSGA-II** (Non-Dominated Sorting Genetic Algorithm II) to automatically discover the Pareto-optimal front. This provides the engineering team with a set of optimal trade-off solutions rather than a single compromise.

---

## This is a the basic overview of how our algorithm works(The basic understanding for reference).

---

First, we randomly generate values within the constraints; these are represented by the vector of variables called chromosomes. Then, these data points are split into ranks based on non-domination. The ranks are sorted such that the first rank is not dominated by any other, and the next rank is only dominated by the first rank, and so on; this forms the Pareto fronts.

Then, we select parents using a Binary Tournament: we take two random points and compare them. Better ranks win; if the ranks are the same, we check the crowding distance, which is calculated to promote diversity (the solution in the less crowded region wins).

Now comes Crossover and Mutation. Crossover is carried out on the selected parents to create children (offspring), and mutation is applied to these children to add randomness.

After that, the parents and children are combined into a single population of size 2N. We sort this combined group using non-dominated sorting. The new generation is formed by filling it with the best fronts (Rank 1, then Rank 2, etc.). If a front cannot fit entirely into the new generation, we sort that specific front by crowding distance and pick the most diverse solutions to fill the remaining slots.

This complete process is repeated until a Termination Criteria is met, and then you will have the best trade-off solutions among the objectives.## 🚀 Getting Started (How to Reproduce)
Follow these steps to replicate our experimental results.

---
## Getting Started. (How can you run the same project in your computer)
Follow the steps below you can run the code in your pc.

### 1. Installation
Clone the repository and install the required dependencies.
git clone <your-repo-url>
cd <your-repo-folder>
pip install -r requirements.txt

(This assumes that you already have python installed in your pc and also pip. Python >= **3.9** and pip >= **21.0**)

## Running the code
To run the code after cloning the repo and installing all the requirements needed run plot_results.py.


## Structure of the Repository
├── PHASES.md               # Detailed Design Log & Engineering Decisions (Evaluation Report)
├── README.md               # Project Overview & Installation Guide
├── requirements.txt        # List of Python dependencies (numpy, pymoo, matplotlib)
├── results/                # Output folder for graphs and data
│   └── pareto_front.png    # Final visualization of the Trade-off Curve
└── src/                    # Source Code
    ├── __init__.py         # Package initializer
    ├── model.py            # The Physics Model (Phase 2): Defines Energy & Acceleration logic
    ├── nsga2_problem.py    # The Optimization Problem Class (Phase 3): Connects Model to Pymoo
    └── run_nsga2.py        # Main Execution Script: Configures and runs the Algorithm