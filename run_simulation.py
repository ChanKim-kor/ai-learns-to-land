from physics import simulate_delayed_ignition

if __name__ == "__main__":
    h0 = 100
    v0 = -10
    mass = 100
    g = 9.81
    dt = 0.3
    ignition_time = 4
    burn_thrust = 2000
    max_time = 10

    result = simulate_delayed_ignition(h0, v0, mass, g, ignition_time, burn_thrust, dt, max_time)
    history = result["history"]

    for elapsed, height, velocity in history:
        print(
                f"Time: {elapsed:.5f} [s]  "
                f"Height: {height:.5f} [m]  "
                f"Velocity: {velocity:.5f} [m/s]"
            )
    print(f"Termination reason: {result["termination_reason"]}")