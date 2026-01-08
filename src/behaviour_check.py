from objective_functions import acceleration_time, energy_consumption

def run_behaviour_check():
    print("=== Behaviour Check===\n")

    # Case 1: High power, low mass (performance-oriented)
    P, M, G = 150, 1000, 4.5
    print("High Performance Setup")
    print("Acceleration:", acceleration_time(P, M, G))
    print("Energy:", energy_consumption(P, M, G))
    print()

    # Case 2: Low power, high mass (efficiency-oriented)
    P, M, G = 60, 1800, 4.5
    print("Low Performance Setup")
    print("Acceleration:", acceleration_time(P, M, G))
    print("Energy:", energy_consumption(P, M, G))
    print()

    # Case 3: Same P and M, different gear ratios
    P, M = 100, 1400
    print("Gear Ratio Comparison")
    for G in [3.0, 4.5, 6.0]:
        print(f"G = {G}")
        print("  Acceleration:", acceleration_time(P, M, G))
        print("  Energy:", energy_consumption(P, M, G))
        print()

if __name__ == "__main__":
    run_behaviour_check()
