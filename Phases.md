# Phase 1: Problem Definition & Requirements

## 1. Engineering Use Case
Scenario: Early-stage passenger vehicle design optimization.
**Problem Statement:** Automotive engineers face a fundamental conflict when designing powertrains:
* Increasing power improves acceleration (performance) but drastically increases energy consumption.
* Reducing mass improves efficiency but may limit structural or powertrain options.
* There is no single "best" car; there is only a set of optimal trade-offs.
* This optimization is performed during the conceptual design phase, where rapid exploration of design trade-offs is prioritized over high-fidelity simulation.

**Solution Approach:** We utilize **NSGA-II (Non-Dominated Sorting Genetic Algorithm II)** to automate the discovery of the Pareto-optimal front. This allows us to present a set of optimal design alternatives (e.g., "High Performance/High Energy" vs. "Eco-Mode/Low Energy") rather than a single compromise solution.

---

## 2. Mathematical Formulation

### 2.1 Decision Variables (The Genotype)
The optimization algorithm controls three core design parameters. These variables were selected to balance computational simplicity with physical plausibility.

| Variable Symbol | Description | Unit | Lower Bound | Upper Bound | Justification |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **P** | Engine/Motor Power | kW | 60 | 150 | Ranges from economy city car to performance sedan. |
| **M** | Vehicle Mass | kg | 1000 | 1800 | Realistic curb weights for modern passenger vehicles. |
| **G** | Final Gear Ratio | - | 3.0 | 6.0 | Typical final drive ratios for standard transmissions. |

### 2.2 Objectives (The Phenotype)
We perform **Minimization** on both objectives.

**Objective 1: Acceleration Time ($f_1$)**
* **Goal:** Minimize time to reach 100 km/h.
* **Description:** Represents the "Performance" aspect of the vehicle.
* **Conflict:** Requires high power and low mass.

**Objective 2: Energy Consumption ($f_2$)**
* **Goal:** Minimize Energy/Fuel used per 100 km.
* **Description:** Represents the "Efficiency" aspect of the vehicle.
* **Conflict:** Penalizes high power and high mass.

Optimization Objective:
Minimize F(x) = [f₁(P, M, G), f₂(P, M, G)]

### 2.3 Constraints
The system puts on the following hard constraints. Any solution violating these is considered invalid.

1.  **Variable Bounds:** $60 \le P \le 150$, $1000 \le M \le 1800$, $3.0 \le G \le 6.0$
2.  **Performance Threshold (Acceptance Criterion):** Ideally, Acceleration time should ideally be ≤ 12.0 seconds for the vehicle to be market-viable. This condition is used post-optimization for solution filtering and validation, not as a hard feasibility constraint.

---

## 3. Algorithm Selection Justification

**Selected Algorithm:** NSGA-II (Non-Dominated Sorting Genetic Algorithm II)

**Why NSGA-II?**
1.  **Multi-Objective Nature:** The problem has two conflicting objectives ($f_1$ vs. $f_2$). Single-objective algorithms (like standard GA or Gradient Descent) cannot capture the trade-off curve.
2.  **Pareto Optimality:** NSGA-II is specifically designed to maintain a diverse set of non-dominated solutions (the Pareto Front).
3.  **Elitism:** The algorithm preserves the best solutions from previous generations, ensuring the population quality never degrades.
4.  **Course Alignment:** This method directly addresses the requirements for "Automotive Case Study #3" as defined in the "Machine Learning in Engineering Applications" curriculum.

---

## 4. Technical Requirements
To reproduce the results of Phase 1 and Phase 2, the following environment is required:

* **Language:** Python 3.8+
* **Core Libraries:**
    * `numpy`: For vectorized physics calculations and matrix operations.
    * `matplotlib`: For visualizing the Pareto Front (Objective Space).
    * `random`: For stochastic processes (Initialization, Mutation).
    * `pymoo`: Implementation of NSGA-II and evolutionary operators.
