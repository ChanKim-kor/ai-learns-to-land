import matplotlib.pyplot as plt
import math
import numpy as np
from matplotlib.animation import FuncAnimation
from physics import simulate_fueled_delayed_ignition


if __name__ == "__main__":
    fig, ax = plt.subplots()

    ax.set_xlim(-10, 10)
    ax.set_ylim(-5, 120)

    ignition_time = 1
    burn_thrust = 2000
    result = simulate_fueled_delayed_ignition(
        h0 = 100,
        v0 = -10,
        dry_mass = 100,
        propellant_mass = 0.5,
        g = 9.81,
        ignition_time = ignition_time,
        burn_thrust = burn_thrust,
        specific_impulse = 200,
        dt = 0.05,
        max_time = 10,
    )

    history = result["history"]
    times = [record[0] for record in history]
    heights = [record[1] for record in history]
    velocities = [record[2] for record in history]
    propellants = [record[3] for record in history]

    # Add the ground and rocket here
    fps = 30
    duration = times[-1]
    frame_count = math.ceil(duration * fps) + 1
    frame_times = np.linspace(0, duration, frame_count)
    frame_interval = duration / (frame_count - 1) * 1000

    ax.axhline(y = 0, color = "black")
    rocket, = ax.plot(
        [0], [heights[0]],
        marker="^",
        markersize = 12,
        linestyle = "None",)

    flame, = ax.plot(
        [0], [100],
        marker = "v",
        color = "orange",
        markersize = 8,
        linestyle = "None",
        visible = False,
    )

    status_text = ax.text(
        0.03,
        0.97,
        "",
        transform=ax.transAxes,
        va = "top"
    )


    def update(frame_time):
        height = np.interp(frame_time, times, heights)
        velocity = np.interp(frame_time, times, velocities)
        propellant = np.interp(frame_time, times, propellants)
        rocket.set_data([0], [height])
        if frame_time >= ignition_time and propellant > 0 and burn_thrust > 0:
            flame.set_visible(True)
            flame.set_data([0], [height - 5])
        else:
            flame.set_visible(False)

        status_text.set_text(
            f"Time: {frame_time:.2f} [s]\n"
            f"Height: {height:.2f} [m]\n"
            f"Velocity: {velocity:.2f} [m/s]\n"
            f"Propellant: {propellant:.2f} [kg]"
        )

    animation = FuncAnimation(
        fig,
        update,
        frames = frame_times,
        interval = frame_interval,
        repeat = False,
    )

    ax.set_title("Vertical rocket View")
    ax.set_ylabel("Height [m]")
    plt.show()