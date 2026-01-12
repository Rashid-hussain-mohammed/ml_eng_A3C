## System Boundaries
This study focuses exclusively on high-level vehicle design parameters affecting
energy consumption and performance. Detailed subsystems such as aerodynamics,
thermal behavior, drivetrain losses, road conditions, and driver behavior are
outside the scope of this work. These effects are implicitly approximated through
simplified objective functions.

## Assumptions
- The vehicle operates under standard driving conditions.
- Energy consumption and acceleration can be approximated using analytical models.
- Effects of aerodynamics and road grade are not modeled explicitly.
- All design variables are independent.


# Project Phases & Design Log (Evaluation Report)

##  Document Scope
This document records the engineering decisions, mathematical formulations, system boundaries, and validation steps taken throughout the project lifecycle. It serves as the formal **Evaluation Report** for Milestone M4.

## 1. System Boundaries & Assumptions
**Boundaries:**
This study focuses exclusively on high-level vehicle design parameters affecting energy consumption and performance. Detailed subsystems such as aerodynamics, thermal behavior, drivetrain losses, and road friction are outside the scope of this work; their effects are implicitly approximated through simplified objective functions.

**Assumptions:**
1.  The vehicle operates under standard driving conditions.
2.  Energy consumption and acceleration can be approximated using analytical models.
3.  Design variables (Mass, Power, Gear Ratio) are treated as independent.

---

## Phase 1: Problem Definition & Requirements

### 1.1 Engineering Use Case
**Scenario:** Early-stage passenger vehicle design optimization.
**Problem Statement:** Automotive engineers face a fundamental conflict:
* Increasing power improves acceleration (performance) but drastically increases energy consumption.
* Reducing mass improves efficiency but may limit structural options.
* **Result:** There is no single "best" car; there is only a set of optimal trade-offs (Pareto Front).

### 1.2 Solution Approach
We utilize **NSGA-II (Non-Dominated Sorting Genetic Algorithm II)** to automate the discovery of these trade-offs. This allows us to present a set of optimal design alternatives (e.g., "Sport Mode" vs. "Eco Mode") rather than a single compromise solution.

### 1.3 Decision Variables (The Genotype)
| Variable | Symbol | Range | Unit | Justification |
| :--- | :--- | :--- | :--- | :--- |
| **Power** | $P$ | 60 - 150 | kW | Ranges from economy city car to performance sedan. |
| **Mass** | $M$ | 1000 - 1800 | kg | Realistic curb weights for modern passenger vehicles. |
| **Gear Ratio** | $G$ | 3.0 - 6.0 | - | Typical final drive ratios for standard transmissions. |

### 1.4 Objectives (The Phenotype)
We perform **Minimization** on both objectives.
1.  **Acceleration Time ($f_1$):** Minimize time to reach 100 km/h.
2.  **Energy Consumption ($f_2$):** Minimize Energy/Fuel used per 100 km.

---

## Phase 2: System Modeling & Simulation

### 2.1 Scope Note (Simplification)
* **Lecture Alignment:** The course case study references complex drivetrain variables (axle ratio, rolling radius).
* **Implementation Decision:** For this prototype (Milestone M2), we utilized a **Simplified Parametric Model**. We mapped the complex physics into normalized functions of Mass, Power, and Gear Ratio. This allows us to demonstrate the **NSGA-II algorithm's behavior** effectively without requiring a high-fidelity physics engine.

### 2.2 Mathematical Strategy: Normalization
To prevent numerical bias (where large numbers like Mass=1800 dominate small numbers like Gear=4), we implemented **Min-Max Normalization**.
* All inputs are converted to a $0 \dots 1$ scale before calculation.
* **Benefit:** This ensures the optimizer treats all variables with equal importance.

### 2.3 Verification (Sanity Check)
Manual boundary tests confirmed physics compliance:
* **High Power / Low Mass:** Resulted in fast acceleration but high energy consumption. (Pass)
* **Low Power / High Mass:** Resulted in slow acceleration but better efficiency. (Pass)

---

## Phase 3: Algorithm Implementation

### 3.1 Tool Selection
We transitioned to the **`pymoo`** Python framework for the production implementation.
* **Justification:** `pymoo` offers a verified, industry-standard implementation of NSGA-II. This eliminates potential bugs in the sorting/crowding logic and allows us to focus on the vehicle physics model.

### 3.2 Algorithm Configuration
* **Population Size:** 120 (High diversity)
* **Generations:** 200 (Ensures convergence)
* **Crossover:** SBX (Simulated Binary Crossover) with $\eta=20$
* **Mutation:** Polynomial Mutation with $\eta=25$

---

## Phase 4: Validation & Results

### 4.1 Pareto Front Analysis
The algorithm successfully produced a convex Pareto Front, confirming the conflicting nature of the objectives.
* **Region A (Top-Left):** "Eco-Cruisers." Low Power, Low Mass. Slow but very efficient.
* **Region B (Bottom-Right):** "Sport Performance." High Power. Fast but energy-intensive.
* **Region C (Middle):** "Balanced Commuters." The optimal compromise for a standard vehicle.

### 4.2 Stability Check
We performed 7 independent runs with different random seeds. The resulting Pareto fronts overlapped consistently, validating that the algorithm is robust and has converged to a global optimum rather than a local anomaly.

### 4.3 Conclusion
The project meets all acceptance criteria defined in Milestone M3. The system successfully automates the trade-off analysis, providing the engineering team with a diverse set of optimal vehicle configurations.