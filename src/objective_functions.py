P_MIN, P_MAX = 60.0, 150.0
M_MIN, M_MAX = 1000.0, 1800.0
G_MIN, G_MAX = 3.0, 6.0

EPSILON = 0.1  # Numerical stability


def normalize(value, vmin, vmax):
    return (value - vmin) / (vmax - vmin)


def gear_efficiency(G_n):
    return 1.0 - (G_n - 0.5) ** 2


def acceleration_time(P, M, G):
    P_n = normalize(P, P_MIN, P_MAX)
    M_n = normalize(M, M_MIN, M_MAX)
    G_n = normalize(G, G_MIN, G_MAX)

    eta_g = gear_efficiency(G_n)

    return 1.0 + M_n / (P_n * eta_g + EPSILON)


def energy_consumption(P, M, G):
    P_n = normalize(P, P_MIN, P_MAX)
    M_n = normalize(M, M_MIN, M_MAX)
    G_n = normalize(G, G_MIN, G_MAX)

    eta_g = gear_efficiency(G_n)

    return 0.5 + M_n * (0.5 + P_n ** 2) * (1.0 + (1.0 - eta_g))

