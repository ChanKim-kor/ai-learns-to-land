import matplotlib.pyplot as plt
import math
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.transforms import ScaledTranslation
from physics import simulate_fueled_delayed_ignition

from time import perf_counter

def animate_flight(result, ignition_time, burn_thrust):
    fig, axes = plt.subplot_mosaic(
        [
            ["rocket", "height"],
            ["rocket", "velocity"],
            ["rocket", "propellant"],
        ],
        figsize = (11, 7),
        layout = "constrained",
    )
    axes["velocity"].sharex(axes["height"])
    axes["propellant"].sharex(axes["height"])


    history = result["history"]
    times = [record[0] for record in history]
    heights = [record[1] for record in history]
    velocities = [record[2] for record in history]
    propellants = [record[3] for record in history]

    fps = 30
    duration = times[-1]
    frame_interval = 1000 / fps
    
    ax_rocket = axes["rocket"]
    ax_rocket.set_xlim(-10, 10)
    ax_rocket.set_ylim(-5, max(120, 1.2 * max(heights)))

    ax_rocket.axhline(y = 0, color = "black")
    rocket, = ax_rocket.plot(
        [0], [heights[0]],
        marker="^",
        markersize = 12,
        linestyle = "None",)

    flame_offset = ScaledTranslation(
        0,
        -12 / 72,
        fig.dpi_scale_trans,
    )

    flame, = ax_rocket.plot(
        [0], [heights[0]],
        marker = "v",
        color = "orange",
        markersize = 8,
        linestyle = "None",
        visible = False,
        transform = ax_rocket.transData + flame_offset
    )

    status_text = ax_rocket.text(
        0.03,
        0.97,
        "",
        transform=ax_rocket.transAxes,
        va = "top"
    )


    frame_wall_times = []

    def playback_times():
        start_time = perf_counter()

        while True:
            elapsed = perf_counter() - start_time
            yield min(elapsed, duration)

            if elapsed >= duration:
                break


    def update(frame_time):
        frame_wall_times.append(perf_counter())
        height = np.interp(frame_time, times, heights)
        velocity = np.interp(frame_time, times, velocities)
        propellant = np.interp(frame_time, times, propellants)
        rocket.set_data([0], [height])
        height_cursor.set_xdata([frame_time, frame_time])
        velocity_cursor.set_xdata([frame_time, frame_time])
        propellant_cursor.set_xdata([frame_time, frame_time])
        
        if frame_time >= ignition_time and propellant > 0 and burn_thrust > 0:
            flame.set_visible(True)
            flame.set_data([0], [height])
        else:
            flame.set_visible(False)

        status_text.set_text(
            f"Time: {frame_time:.2f} [s]\n"
            f"Height: {height:.2f} [m]\n"
            f"Velocity: {velocity:.2f} [m/s]\n"
            f"Propellant: {propellant:.2f} [kg]"
        )

        if frame_time >= duration:
            playback_elapsed = frame_wall_times[-1] - frame_wall_times[0]
            print(f"Simulation duration: {duration:.3f} [s]")
            print(f"Playback elapsed: {playback_elapsed:.3f} [s]")
        return rocket, flame, status_text, height_cursor, velocity_cursor, propellant_cursor
        

    ax_rocket.set_title("Vertical rocket View")
    ax_rocket.set_ylabel("Height [m]")

    axes["height"].plot(times, heights)
    axes["height"].set_ylabel("Height [m]")
    axes["velocity"].plot(times, velocities)
    axes["velocity"].set_ylabel("Velocity [m/s]")
    axes["propellant"].plot(times, propellants)
    axes["propellant"].set_ylabel("Propellant [kg]")
    axes["propellant"].set_xlabel("Time [s]")

    height_cursor = axes["height"].axvline(
        x = 0,
        color = "red",
        linestyle = "--",
    )

    velocity_cursor = axes["velocity"].axvline(
        x = 0,
        color = "red",
        linestyle = "--",
    )

    propellant_cursor = axes["propellant"].axvline(
        x = 0,
        color = "red",
        linestyle = "--",
    )

    # Calculate the layout once and keep it fixed
    fig.canvas.draw()
    fig.set_layout_engine("none")


    def init_animation():
        return rocket, flame, status_text, height_cursor, velocity_cursor, propellant_cursor
    

    animation = FuncAnimation(
        fig,
        update,
        frames = playback_times,
        init_func = init_animation,
        interval = frame_interval,
        repeat = False,
        blit = True,
        cache_frame_data = False,
    )

    plt.show()

if __name__ == "__main__":
    ignition_time = 1
    burn_thrust = 2000
    result = simulate_fueled_delayed_ignition(
        h0 = 100,
        v0 = 0,
        dry_mass = 100,
        propellant_mass = 15,
        g = 9.81,
        ignition_time = ignition_time,
        burn_thrust = burn_thrust,
        specific_impulse = 200,
        dt = 0.05,
        max_time = 30,
    )

    animate_flight(result, ignition_time, burn_thrust)