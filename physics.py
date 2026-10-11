import math


def free_fall_velocity(v0, g, elapsed):
    if g < 0:
        raise ValueError("Gravity magnitude must be non-negative.")
    if elapsed < 0:
        raise ValueError("Elapsed time must be non-negative.")
    
    return v0 - g * elapsed

def free_fall_height(h0, v0, g, elapsed):
    if h0 < 0:
        raise ValueError("Height must be non-negative.")
    if g < 0:
        raise ValueError("Gravity magnitude must be non-negative.")
    if elapsed < 0:
        raise ValueError("Elapsed time must be non-negative.")

    return h0 + v0 * elapsed - 0.5 * g * (elapsed ** 2)

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