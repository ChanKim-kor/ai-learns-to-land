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
        / "final_model.pt"
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

    animate_flight(result)