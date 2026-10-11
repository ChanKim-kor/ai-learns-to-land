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

def simulate_free_fall(h0, v0, g, dt, max_time=20):
    elapsed = 0
    history = []

    if max_time < 0:
        raise ValueError("Simulation time must be non-negative.")
    if dt <= 0:
        raise ValueError("dt must be a positive value.")
    
    while True:
        current_h = free_fall_height(h0, v0, g, elapsed)
        current_v = free_fall_velocity(v0, g, elapsed)
        history.append((elapsed, current_h, current_v))
        if current_h <= 0 or elapsed >= max_time:
            break
        elapsed = min(max_time, elapsed + dt)

    return history

h0 = 100
v0 = -10
g = 9.81
t = 0
dt = 0.01

history = simulate_free_fall(h0, v0, g, dt)

for t, height, velocity in history:
    print(
        f"Time: {t:.2f} [s]  "
        f"Height: {height:.4f} [m]  "
        f"Velovity: {velocity:.4f} [m/s]"
    )