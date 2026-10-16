import math
from physics import simulate_fueled_delayed_ignition, propellant_mass_flow_rate


h0 = 1000
v0 = 0
dry_mass = 100
propellant_mass = 20
initial_mass = dry_mass + propellant_mass
g = 9.81
ignition_time = 0
thrust = 2000
specific_impulse = 200
q = propellant_mass_flow_rate(thrust, specific_impulse)
max_time = 3
mass_at_end = initial_mass - q * max_time
dt_list = (0.3, 0.15, 0.075)

reference_velocity = (
    v0
    + thrust / q * math.log(initial_mass / (initial_mass - q * max_time))
    - g * max_time
    )
reference_height = (
    h0 + v0 * max_time
    + thrust / q * (max_time - mass_at_end / q * math.log(initial_mass / mass_at_end))
    - 1 / 2 * g * max_time ** 2
)

previous_height_error = None
height_error_ratio = None
previous_velocity_error = None
velocity_error_ratio = None

for dt in dt_list:
    result = simulate_fueled_delayed_ignition(h0, v0, dry_mass, propellant_mass,
                                              g, ignition_time, thrust, specific_impulse,
                                              dt, max_time)
    last_time, last_height, last_velocity, last_propellant = result["history"][-1]

    height_error = abs(reference_height - last_height)
    velocity_error = abs(reference_velocity - last_velocity)
    if previous_velocity_error is not None:
        velocity_error_ratio = previous_velocity_error / velocity_error
    if previous_height_error is not None:
        height_error_ratio = previous_height_error / height_error

    height_error_ratio_text = (
        f"{height_error_ratio:.5f}"
        if height_error_ratio is not None
        else "None"
    )
    velocity_error_ratio_text = (
        f"{velocity_error_ratio:.5f}"
        if velocity_error_ratio is not None
        else "None"
    )

    print(f"dt = {dt:.3f}\n"
          f"Last time: {last_time:.5f} [s]\n"
          f"Last height: {last_height:.5f} [m]\n"
          f"Height error: {height_error:.5f} [m] | Height error ratio = {height_error_ratio_text}\n"
          f"Last velocity: {last_velocity:.5f} [m/s]\n"
          f"Velocity error: {velocity_error:.5f} [m/s] | Velocity error ratio = {velocity_error_ratio_text}\n"
          f"Last propellant mass = {last_propellant:.5f} [kg]\n"
          )

    previous_height_error = height_error
    previous_velocity_error = velocity_error