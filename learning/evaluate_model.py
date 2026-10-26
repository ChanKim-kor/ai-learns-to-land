from pathlib import Path

import torch
from torch import nn

from learning.train_rocket_transition import run_episode
from visualize_simulation import animate_flight

if __name__ == "__main__":
    model = nn.Sequential(
        nn.Linear(4, 4, dtype=torch.float64),
        nn.ReLU(),
        nn.Linear(4, 2, dtype=torch.float64),
    )

    model_path = (
        Path(__file__).resolve().parents[1]
        / "runs"
        / "best_model.pt"
    )
    print(f"Loaded model: {model_path.name}")
    state = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state)
    model.eval()

    result = run_episode(
        model,
        optimizer=None,
        replay_buffer=None,
        epsilon=0.0,
        target_model=None,
        training=False,
        record_history=True
    )

    print(
        f"Status: {result['status']}\n"
        f"Final velocity: {result['final_velocity']:.3f} [m/s]\n"
        f"Recorded states: {len(result['history'])}\n"
        f"Recorded control intervals: {len(result['control_history'])}"
    )

    previous_thrust = None

    for index in range(len(result["control_history"])):
        start, end, thrust = result["control_history"][index]

        if thrust != previous_thrust:
            elapsed, height, velocity, propellant = (
                result["history"][index]
            )
            engine_status = "ON" if thrust > 0 else "OFF"

            print(
                f"Time: {start:.3f} [s] | "
                f"Engine: {engine_status} | "
                f"Height: {height:.3f} [m] | "
                f"Velocity: {velocity:.3f} [m/s] | "
                f"Propellant: {propellant:.3f} [kg]"
            )
        previous_thrust = thrust
    animate_flight(result)