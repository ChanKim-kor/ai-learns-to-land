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

def simulate_constant_acceleration(h0, v0, acceleration, dt, max_time=20):
    elapsed = 0
    history = []

    if max_time < 0:
        raise ValueError("Simulation time must be non-negative.")
    if dt <= 0:
        raise ValueError("dt must be a positive value.")

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

def simulate_constant_thrust(h0, v0, thrust, mass, g, dt, max_time=20):
    acceleration = vertical_acceleration(thrust, mass, g)
    result = simulate_constant_acceleration(h0, v0, acceleration, dt, max_time=max_time)

    return result

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
    if g < 0:
        raise ValueError("Gravity magnitude must be non-negative.")
    result = simulate_constant_acceleration(h0, v0, -g, dt, max_time=max_time)

    return result

def advance_vertical_state(height, velocity, acceleration, dt):
    if dt <= 0:
        raise ValueError("Simulation time step must be a positive value")
    h_next = constant_acceleration_height(height, velocity, acceleration, dt)
    v_next = constant_acceleration_velocity(velocity, acceleration, dt)

    return h_next, v_next

def delayed_ignition_thrust(elapsed, ignition_time, burn_thrust):
    if elapsed < 0:
        raise ValueError("Elapsed time must be non-negative.")
    if ignition_time < 0:
        raise ValueError("Ignition time must be non-negative.")
    if burn_thrust < 0:
        raise ValueError("Thrust force must be non-negative.")

    if elapsed < ignition_time:
        return 0
    else:
        return burn_thrust

def simulate_delayed_ignition(h0, v0, mass, g, ignition_time, burn_thrust, dt, max_time=20):   
    if h0 < 0:
        raise ValueError("Height must be non-negative")
    if mass <= 0:
        raise ValueError("Mass must be positive ")
    if ignition_time < 0:
        raise ValueError("Ignition time must be non-negative")
    if burn_thrust < 0:
        raise ValueError("Thrust force must be non-negative")
    if max_time < 0:
        raise ValueError("Simulation time must be non-negative")
    if dt <= 0:
        raise ValueError("Simulation time step must be positive")
    if g < 0:
        raise ValueError("Gravity magnitude must be non-negative")
    
    height = h0
    velocity = v0
    ground_contact = False
    elapsed = 0
    history = [(elapsed, height, velocity)] # Initial state

    while True:
        if height <= 0:
            ground_contact = True
            break
        if elapsed >= max_time or ground_contact:
            break
        thrust = delayed_ignition_thrust(elapsed, ignition_time, burn_thrust)
        acceleration = vertical_acceleration(thrust, mass, g)

        next_time = min(elapsed + dt, max_time)
        if elapsed < ignition_time:
            next_time = min(next_time, ignition_time)
        step_dt = next_time - elapsed

        contact_dt = constant_acceleration_contact_time(height, velocity, acceleration)
        if contact_dt is not None and contact_dt <= step_dt:
            step_dt = contact_dt
            next_time = elapsed + step_dt
            ground_contact = True
        
        height, velocity = advance_vertical_state(height, velocity, acceleration, step_dt)
        elapsed = next_time

        history.append((elapsed, height, velocity))
        

    if ground_contact:
        termination_reason = "ground_contact"
    else:
        termination_reason = "time_limit"

    return {
        "history": history,
        "termination_reason": termination_reason
    }

def propellant_mass_flow_rate(thrust, specific_impulse, g0=9.80665):
    if thrust < 0:
        raise ValueError("Thrust must be non-negative.")
    if specific_impulse <= 0:
        raise ValueError("Specific impulse must be positive.")
    if g0 <= 0:
        raise ValueError("g0 must be positive.")

    q = thrust / (specific_impulse * g0)
    return q

def propellant_depletion_time(propellant_mass, mass_flow_rate):
    if propellant_mass < 0:
        raise ValueError("Propellant mass must be non-negative")
    if mass_flow_rate < 0:
        raise ValueError("Propellant mass flow rate must be non-negative")

    if propellant_mass == 0:
        return 0
    if mass_flow_rate == 0:
        return None

    return propellant_mass / mass_flow_rate

def advance_burning_state(height, velocity, propellant_mass, dry_mass, thrust, specific_impulse, g, dt):
    if dry_mass <= 0:
        raise ValueError("Dry mass must be non-negative")
    if propellant_mass <= 0:
        raise ValueError("Propellant mass must be positive")
    total_mass = dry_mass + propellant_mass
    q = propellant_mass_flow_rate(thrust, specific_impulse)
    depletion_time = propellant_depletion_time(propellant_mass, q)
    if depletion_time is not None and dt > depletion_time:
        raise ValueError("Time step is longer than propellant depletion time")
    acceleration = vertical_acceleration(thrust, total_mass, g)
    next_height, next_velocity = advance_vertical_state(height, velocity, acceleration, dt)
    remaining_propellant_mass = propellant_mass - q * dt

    return next_height, next_velocity, remaining_propellant_mass

def simulate_fueled_delayed_ignition(h0, v0, dry_mass, 
                                     propellant_mass, g, ignition_time, burn_thrust, 
                                     specific_impulse, dt, max_time=20):
    if h0 < 0:
        raise ValueError("Height must be non-negative")
    if g < 0:
        raise ValueError("g must be non-negative.")
    if dt <= 0:
        raise ValueError("Simulation time step must be positive.")
    if max_time < 0:
        raise ValueError("Max simulation time must be non-negative.")
    if dry_mass <= 0:
        raise ValueError("Dry mass must be positive.")
    if propellant_mass < 0:
        raise ValueError("Propellant mass must be non-negative.")
    if specific_impulse <= 0:
        raise ValueError("Specific impulse must be positive.")
    if ignition_time < 0:
        raise ValueError("Ignition time must be non-negative.")
    if burn_thrust < 0:
        raise ValueError("Thrust force must be non-negative.")

    height = h0
    velocity = v0
    remaining_propellant = propellant_mass
    elapsed = 0
    ground_contact = False

    history = [(elapsed, height, velocity, remaining_propellant)] # Initial state

    while True:
        if height <= 0 or ground_contact:
            ground_contact = True
            termination_reason = "ground_contact"
            break
        if elapsed >= max_time:
            termination_reason = "time_limit"
            break

        fuel_depleted = False
        commanded_thrust = delayed_ignition_thrust(elapsed, ignition_time, burn_thrust)
        actual_thrust = commanded_thrust if remaining_propellant > 0 else 0
        total_mass = dry_mass + remaining_propellant
        acceleration = vertical_acceleration(actual_thrust, total_mass, g)

        q = propellant_mass_flow_rate(actual_thrust, specific_impulse)

        next_time = min(elapsed + dt, max_time)

        if elapsed < ignition_time:
            next_time = min(next_time, ignition_time)

        if q > 0:
            depletion_dt = propellant_depletion_time(remaining_propellant, q)
            depletion_time = elapsed + depletion_dt

            if depletion_time <= next_time:
                next_time = depletion_time
                fuel_depleted = True

        step_dt = depletion_dt if fuel_depleted else next_time - elapsed

        contact_dt = constant_acceleration_contact_time(height, velocity, acceleration)
        if contact_dt is not None and contact_dt <= step_dt:
            if contact_dt < step_dt:
                fuel_depleted = False
            next_time = elapsed + contact_dt
            step_dt = contact_dt
            ground_contact = True


        if actual_thrust > 0:
            height, velocity, remaining_propellant = advance_burning_state(height, velocity, remaining_propellant,
                                                                           dry_mass, actual_thrust, specific_impulse,
                                                                           g, step_dt)
        else:
            height, velocity = advance_vertical_state(height, velocity, acceleration, step_dt)

        elapsed = next_time
        if fuel_depleted: remaining_propellant = 0
        history.append((elapsed, height, velocity, remaining_propellant))


    return{
        "history": history,
        "termination_reason": termination_reason
    }

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


    # result = simulate_free_fall(h0, v0, g, dt)

    # for t, height, velocity in result["history"]:
    #     print(
    #         f"Time: {t:.5f} [s]  "
    #         f"Height: {height:.5f} [m]  "
    #         f"Velocity: {velocity:.5f} [m/s]"
    #     )

    # print(f"Termination reason: {result['termination_reason']}")