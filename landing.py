def classify_landing(velocity, ground_contact, max_landing_speed=2.0):
    if max_landing_speed < 0:
        raise ValueError("Maximum landing speed must be non-negative.")
    if not ground_contact:
        return "in_flight"
    if abs(velocity) <= max_landing_speed:
        return "landed"
    else:
        return "crashed"
    