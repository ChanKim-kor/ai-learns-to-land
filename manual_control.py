import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.transforms import ScaledTranslation
from time import perf_counter
from physics import step_fueled_flight
from landing import classify_landing



if __name__ == "__main__":
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.25)
    lower_limit = -5
    upper_limit = 120
    ax.set_xlim(-10, 10)
    ax.set_ylim(lower_limit, upper_limit)
    ax.axhline(y=0, color="black")
    ax.set_ylabel("Height [m]")
    ax.set_title("Manual Rocket Control")

    controls = {"engine_on": False}
    initial_state = {
        "elapsed": 0.0,
        "height": 100.0,
        "velocity": -10.0,
        "propellant_mass": 15.0,
        "ground_contact": False,
        "finished": False,

    }
    state = initial_state.copy()
    dry_mass = 100
    burn_thrust = 2000
    specific_impulse = 200
    g = 9.81
    physics_dt = 0.02
    max_time = 30

    rocket, = ax.plot(
        [0],
        [state["height"]],
        marker="^",
        markersize=12,
        linestyle="None",
    )

    flame_offset = ScaledTranslation(
        0,
        -12 / 72,
        fig.dpi_scale_trans,
    )

    flame, = ax.plot(
        [0],
        [state["height"]],
        marker="v",
        color="orange",
        markersize=8,
        linestyle="None",
        visible=False,
        transform=ax.transData + flame_offset
    )

    def advance_simulation(target_time):
        target_time = min(target_time, max_time)

        while state["elapsed"] < target_time and not state["finished"]:
            next_time = min(state["elapsed"] + physics_dt, target_time)
            requested_dt = next_time - state["elapsed"]

            commanded_thrust = burn_thrust if controls["engine_on"] else 0
            result = step_fueled_flight(
                height=state["height"],
                velocity=state["velocity"],
                dry_mass=dry_mass,
                propellant_mass=state["propellant_mass"],
                commanded_thrust=commanded_thrust,
                specific_impulse=specific_impulse,
                g=g,
                dt=requested_dt
            )

            if requested_dt == result["elapsed"]:
                state["elapsed"] = next_time
            else:
                state["elapsed"] += result["elapsed"]

            state["height"] = result["height"]
            state["velocity"] = result["velocity"]
            state["propellant_mass"] = result["propellant_mass"]
            state["ground_contact"] = result["ground_contact"]
            state["finished"] = (result["ground_contact"] or state["elapsed"] >= max_time)

    def on_tick():
        target_time = perf_counter() - clock["start_time"]
        advance_simulation(target_time)
        refresh_display()
        if state["finished"]:
            timer.stop()

    def refresh_display():
        rocket.set_ydata([state["height"]])
        engine_active = (
            controls["engine_on"]
            and state["propellant_mass"] > 0
            and burn_thrust > 0
            and not state["finished"]
        )
        flame.set_ydata([state["height"]])
        flame.set_visible(engine_active)
        lower_limit, upper_limit = ax.get_ylim()
        if state["height"] > upper_limit * 0.9:
            new_upper_limit = max(
                upper_limit * 1.5, state['height'] * 1.2,
            )
            ax.set_ylim(lower_limit, new_upper_limit)
        engine_command = "ON" if controls["engine_on"] else "OFF"
        flight_status = classify_landing(
            velocity=state["velocity"],
            ground_contact=state["ground_contact"],
            max_landing_speed=2)
        if state["elapsed"] >= max_time and flight_status == "in_flight":
            flight_status = "time_limit"
        status_text.set_text(
            f"Time: {state["elapsed"]:.2f} [s]\n"
            f"Height: {state["height"]:.2f} [m]\n"
            f"Velocity: {state['velocity']:.2f} [m/s]\n"
            f"Propellant_mass: {state['propellant_mass']:.2f} [kg]\n"
            f"Engine Command: {engine_command}\n"
            f"Flight status: {flight_status}"
        )
        fig.canvas.draw_idle()

    def reset_flight(event):
        timer.stop()
        state.update(initial_state)
        controls["engine_on"] = False
        ax.set_ylim(-5, 120)
        refresh_display()
        clock["start_time"] = perf_counter()
        timer.start()

    status_text = ax.text(
        0.03,
        0.97,
        "Engine command: OFF",
        transform=ax.transAxes,
        va="top",
    )

    def turn_engine_on(event):
        advance_simulation(perf_counter() - clock["start_time"])
        if not state["finished"]:
            controls["engine_on"] = True
        refresh_display()

    def turn_engine_off(event):
        advance_simulation(perf_counter() - clock["start_time"])
        if not state["finished"]:
            controls["engine_on"] = False
        refresh_display()

    on_axes = fig.add_axes([0.10, 0.05, 0.25, 0.075])
    off_axes = fig.add_axes([0.39, 0.05, 0.25, 0.075])
    restart_axes = fig.add_axes([0.68, 0.05, 0.22, 0.075])

    on_button = Button(on_axes, "Engine ON")
    off_button = Button(off_axes, "Engine Off")
    restart_button = Button(restart_axes, "RESTART")
    on_button.on_clicked(turn_engine_on)
    off_button.on_clicked(turn_engine_off)
    restart_button.on_clicked(reset_flight)


    refresh_display()
    fig.canvas.draw()

    timer = fig.canvas.new_timer(interval=33)
    timer.add_callback(on_tick)

    clock = {"start_time": perf_counter()}
    timer.start()
    plt.show()