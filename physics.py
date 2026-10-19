import math

# Constant-acceleration physics
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

def advance_vertical_state(height, velocity, acceleration, dt):
    if dt <= 0:
        raise ValueError("Simulation time step must be a positive value")
    h_next = constant_acceleration_height(height, velocity, acceleration, dt)
    v_next = constant_acceleration_velocity(velocity, acceleration, dt)

    return h_next, v_next


# Propellant and burning-state updates
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


# Thrust schedule

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


# Simulation Runner

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

def simulate_free_fall(h0, v0, g, dt, max_time=20):
    if g < 0:
        raise ValueError("Gravity magnitude must be non-negative.")
    result = simulate_constant_acceleration(h0, v0, -g, dt, max_time=max_time)

    return result

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

def simulate_fueled_delayed_ignition(h0, v0, dry_mass, 
                                     propellant_mass, g, ignition_time, burn_thrust, 
                                     specific_impulse, dt, max_time=20):
    """Simulate vertical motion with delayed ignition and finite propellant

    Use SI units. Each history record contains:
    (time, height, velocity, remaining_propellant).

    Approximate motion by holding acceleration constant within each step,
    using the mass at the start of that step. Shorten steps at ignition,
    propellant depletion, estimated ground contact, and the time limit.

    Propellant depletion turns off the engine but does not end the simulation.

    Return a dictionary containing history and termination_reason.
    Ground contact takes priority when it coincides with the time limit.
    """

    # Validate inputs before handling immediate termination.
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

        # Choose the thrust command and limit the step at scheduled events.
        commanded_thrust = delayed_ignition_thrust(elapsed, ignition_time, burn_thrust)
        next_time = min(elapsed + dt, max_time)
        if elapsed < ignition_time:
            next_time = min(next_time, ignition_time)
        requested_dt = next_time - elapsed
        step_result = step_fueled_flight(height=height,
                                         velocity=velocity,
                                         dry_mass=dry_mass,
                                         propellant_mass=remaining_propellant,
                                         commanded_thrust=commanded_thrust,
                                         specific_impulse=specific_impulse,
                                         g=g, dt=requested_dt)
        height = step_result["height"]
        velocity = step_result["velocity"]
        remaining_propellant = step_result["propellant_mass"]
        ground_contact = step_result["ground_contact"]

        if step_result["elapsed"] == requested_dt:
            elapsed = next_time
        else:
            elapsed += step_result["elapsed"]

        history.append((elapsed, height, velocity, remaining_propellant))


    return{
        "history": history,
        "termination_reason": termination_reason
    }

def step_fueled_flight(
        height,
        velocity,
        dry_mass,
        propellant_mass,
        commanded_thrust,
        specific_impulse,
        g,
        dt,
):
    if height < 0:
        raise ValueError("Height must be non-negative.")
    if dry_mass <= 0:
        raise ValueError("Dry_mass must be positive.")
    if propellant_mass < 0:
        raise ValueError("Propellant must be non-negative.")
    if commanded_thrust < 0:
        raise ValueError("Thrust must be non-negative.")
    if specific_impulse <= 0:
        raise ValueError("Specific impulse must be positive")
    if g < 0:
        raise ValueError("Gravity magnitude must be non-negative")
    if dt <= 0:
        raise ValueError("Simulation time step must be positive")

    total_mass = dry_mass + propellant_mass
    actual_thrust = commanded_thrust if propellant_mass > 0 else 0
    q = propellant_mass_flow_rate(actual_thrust, specific_impulse)

    step_dt = dt
    depleted_this_step = False
    ground_contact = False
    if q > 0:
        depletion_dt = propellant_depletion_time(propellant_mass, q)
        if depletion_dt <= step_dt:
            step_dt = depletion_dt
            depleted_this_step = True

    acceleration = vertical_acceleration(actual_thrust, total_mass, g=g)
    contact_dt = constant_acceleration_contact_time(height, velocity, acceleration)
    if contact_dt is not None and contact_dt <= step_dt:
        if contact_dt < step_dt:
            depleted_this_step = False
        step_dt = contact_dt
        ground_contact = True

    if step_dt > 0:
        if actual_thrust > 0:
            height, velocity, propellant_mass = advance_burning_state(height, velocity, propellant_mass,
                                                                      dry_mass, actual_thrust,
                                                                      specific_impulse, g, step_dt)
        else:
            height, velocity = advance_vertical_state(height, velocity, acceleration, step_dt)

    if depleted_this_step:
        propellant_mass = 0

    return {
        "height": height,
        "velocity": velocity,
        "propellant_mass": propellant_mass,
        "elapsed": step_dt,
        "ground_contact": ground_contact,
        "fuel_depleted": propellant_mass == 0,
    }
    
# Free-fall convenience function

def free_fall_velocity(v0, g, elapsed):
    if g < 0:
        raise ValueError("Gravity magnitude must be non-negative.")
    
    return constant_acceleration_velocity(v0, -g, elapsed)
