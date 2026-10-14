from physics import simulate_fueled_delayed_ignition

if __name__ == "__main__":
    h0 = 100
    v0 = -10
    dry_mass = 100
    propellant_mass = 0.5
    specific_impulse = 200
    ignition_time = 1
    burn_thrust = 2000
    g = 9.81
    dt = 0.3
    max_time = 10

    result = simulate_fueled_delayed_ignition(
        h0=h0,
        v0=v0,
        dry_mass=dry_mass,
        propellant_mass=propellant_mass,
        g=g,
        ignition_time=ignition_time,
        burn_thrust=burn_thrust,
        specific_impulse=specific_impulse,
        dt=dt,
        max_time=max_time,
    )

    history = result["history"]

    for elapsed, height, velocity, propellant in history:
        print(
                f"Time: {elapsed:.5f} [s]  "
                f"Height: {height:.5f} [m]  "
                f"Velocity: {velocity:.5f} [m/s]  "
                f"Propellant: {propellant:.5f} [kg]"
            )
    print(f"Termination reason: {result["termination_reason"]}")