import numpy as np
from pymoo.core.problem import ElementwiseProblem


class VehicleOptimizationProblem(ElementwiseProblem):

    def __init__(self):
        super().__init__(
            n_var=3,              # [Mass, Power, Drag Coefficient]
            n_obj=2,              # Energy, Acceleration Time
            n_constr=0,
            xl=np.array([1000.0, 60.0, 0.25]),
            xu=np.array([1800.0, 150.0, 0.40])
        )

    def _evaluate(self, x, out, *args, **kwargs):
        m, P, Cd = x

        # ---- Acceleration Model (Performance) ----
        accel_time = (m / P) * (1 + 0.05 * Cd)

        # ---- Energy Consumption Model (Efficiency) ----
        drag_energy = Cd * (P ** 1.3) / (m ** 0.3)
        rolling_energy = 0.0004 * (m ** 1.1)

        # Engine efficiency curve
        efficiency = np.exp(-((P - 110) ** 2) / (2 * 35 ** 2))

        energy = (drag_energy + rolling_energy) / efficiency

        # Mild physical uncertainty
        noise = np.random.normal(1.0, 0.02)

        out["F"] = [
            energy * noise,
            accel_time
        ]
