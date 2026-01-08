# Multi-Objective Optimization of Vehicle Design (NSGA-II) (Our Project)

## 📌 Project Overview
**Goal:** Optimize passenger vehicle design to balance two conflicting objectives: **minimizing fuel consumption** and **maximizing acceleration performance**.
**Method:** We utilize the **NSGA-II** evolutionary algorithm to find the Pareto-optimal front, providing a set of best trade-off solutions rather than a single compromise.

---

## This is a the basic overview of how our algorithm works(The basic understanding for reference).

---

First, we randomly generate values within the constraints; these are represented by the vector of variables called chromosomes. Then, these data points are split into ranks based on non-domination. The ranks are sorted such that the first rank is not dominated by any other, and the next rank is only dominated by the first rank, and so on; this forms the Pareto fronts.

Then, we select parents using a Binary Tournament: we take two random points and compare them. Better ranks win; if the ranks are the same, we check the crowding distance, which is calculated to promote diversity (the solution in the less crowded region wins).

Now comes Crossover and Mutation. Crossover is carried out on the selected parents to create children (offspring), and mutation is applied to these children to add randomness.

After that, the parents and children are combined into a single population of size 2N. We sort this combined group using non-dominated sorting. The new generation is formed by filling it with the best fronts (Rank 1, then Rank 2, etc.). If a front cannot fit entirely into the new generation, we sort that specific front by crowding distance and pick the most diverse solutions to fill the remaining slots.

This complete process is repeated until a Termination Criteria is met, and then you will have the best trade-off solutions among the objectives.