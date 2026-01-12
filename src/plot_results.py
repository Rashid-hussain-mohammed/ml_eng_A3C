import matplotlib.pyplot as plt
from run_nsga2 import run_optimization


def plot_multiple_pareto_fronts(n_runs=5):
    plt.figure(figsize=(8, 6))

    for seed in range(1, n_runs + 1):
        res = run_optimization(seed=seed)
        F = res.F

        plt.scatter(
            F[:, 0],
            F[:, 1],
            alpha=0.35,      # transparency
            s=20,
            label=f"Run {seed}"
        )

    plt.xlabel("Energy Consumption (normalized)")
    plt.ylabel("0–100 km/h Acceleration Time (s)")
    plt.title("Pareto Fronts over Multiple NSGA-II Runs")
    plt.grid(True)

    # Optional: comment out legend if cluttered
    # plt.legend()

    plt.tight_layout()
    plt.savefig("results/pareto_front_multiple_runs.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    plot_multiple_pareto_fronts(n_runs=7)
