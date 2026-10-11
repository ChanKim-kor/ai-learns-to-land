import math


def vertical_acceleration(thrust, mass, g):
    if thrust < 0:
        raise ValueError("Thrust value must be non-negative.")
    if mass <= 0:
        raise ValueError("mass must be positive.")
    if g < 0:
        raise ValueError("Gravity magnitude must be non-negative.")

    return thrust / mass - g

def constant_acceleration_velocity(v0, acceleration, elapsed):
    if elapsed < 0:
        raise ValueError("Elapsed time must be non-negative.")
    return v0 + acceleration * elapsed

def constant_acceleration_height(h0, v0, acceleration, elapsed):
    if h0 < 0:
        raise ValueError("Height must be non-negative.")
    if elapsed < 0:
        raise ValueError("Elapsed time must be non-negative.")

    return h0 + v0 * elapsed + 0.5 * acceleration * (elapsed ** 2)

def constant_thrust_velocity(v0, thrust, mass, g, elapsed):
    acceleration = vertical_acceleration(thrust, mass, g)
    return constant_acceleration_velocity(v0, acceleration, elapsed)

def constant_thrust_height(h0, v0, thrust, mass, g, elapsed):
    acceleration = vertical_acceleration(thrust, mass, g)
    return constant_acceleration_height(h0, v0, acceleration, elapsed)

def constant_acceleration_contact_time(h0, v0, acceleration):
    if h0 < 0:
        raise ValueError("Height must be non-negative")
    
    if h0 == 0:
        return 0
    if acceleration == 0:
        if v0 >= 0:
            return None
        else:
            return  -h0 / v0
    else:
        discriminant = v0 ** 2 - 2 * acceleration * h0
        if discriminant < 0:
            return None
        else:
            root_list = [(-v0 + math.sqrt(discriminant)) / acceleration, (-v0 - math.sqrt(discriminant)) / acceleration]
            root_list = [i for i in root_list if i >= 0]
            if len(root_list) == 0:
                return None
            else:
                return min(root_list)

def constant_thrust_contact_time(h0, v0, thrust, mass, g):
    acceleration = vertical_acceleration(thrust, mass, g)
    return constant_acceleration_contact_time(h0, v0, acceleration)

def simulate_constant_thrust(h0, v0, thrust, mass, g, dt, max_time=20):
    elapsed = 0
    history = []

    if max_time < 0:
        raise ValueError("Simulation time must be non-negative.")
    if dt <= 0:
        raise ValueError("dt must be a positive value.")

    acceleration = vertical_acceleration(thrust, mass, g)
    contact_time = constant_acceleration_contact_time(h0, v0, acceleration)

    if contact_time is not None and contact_time <= max_time:
        end_time = contact_time
        termination_reason = "ground_contact"
    else:
        end_time = max_time
        termination_reason = "time_limit"

    while True:
        current_h = constant_acceleration_height(h0, v0, acceleration, elapsed)
        current_v = constant_acceleration_velocity(v0, acceleration, elapsed)
        history.append((elapsed, current_h, current_v))

        if elapsed >= end_time:
            break
        elapsed = min(elapsed + dt, end_time)

    return {
        "history": history,
        "termination_reason": termination_reason
    }

def free_fall_velocity(v0, g, elapsed):
    if g < 0:
        raise ValueError("Gravity magnitude must be non-negative.")
    
    return constant_acceleration_velocity(v0, -g, elapsed)

def free_fall_height(h0, v0, g, elapsed):
    if g < 0:
        raise ValueError("Gravity magnitude must be non-negative.")

    return constant_acceleration_height(h0, v0, -g, elapsed)

def free_fall_contact_time(h0, v0, g):
    if h0 < 0:
        raise ValueError("Height must be non-negative.")
    if g < 0:
        raise ValueError("Gravity magnitude must be non-negative.")

    if h0 == 0:
        return 0
    if g > 0:
        return (v0 + math.sqrt(v0 ** 2 + 2 * g * h0))/g
    else: # g = 0
        if v0 < 0:
            return -h0 / v0
        else:
            return None


def simulate_free_fall(h0, v0, g, dt, max_time=20):
    elapsed = 0
    history = []

    if max_time < 0:
        raise ValueError("Simulation time must be non-negative.")
    if dt <= 0:
        raise ValueError("dt must be a positive value.")

    contact_time = free_fall_contact_time(h0, v0, g)
    if contact_time is not None and contact_time <= max_time:
        end_time = contact_time
        termination_reason = "ground_contact"
    else:
        end_time = max_time
        termination_reason = "time_limit"


    while True:
        current_h = free_fall_height(h0, v0, g, elapsed)
        current_v = free_fall_velocity(v0, g, elapsed)
        history.append((elapsed, current_h, current_v))

        if elapsed >= end_time:
            break
        elapsed = min(end_time, elapsed + dt)

    return {
        "history": history,
        "termination_reason": termination_reason
    }

h0 = 100
v0 = -10
g = 9.81
t = 0
dt = 0.1

result = simulate_free_fall(h0, v0, g, dt)

for t, height, velocity in result["history"]:
    print(
        f"Time: {t:.5f} [s]  "
        f"Height: {height:.5f} [m]  "
        f"Velocity: {velocity:.5f} [m/s]"
    )

print(f"Termination reason: {result['termination_reason']}")