import torch
from torch import nn
import random
from collections import deque
from copy import deepcopy
import csv
from pathlib import Path
from physics import step_fueled_flight
from landing import classify_landing

def select_action(model, observation, epsilon):
    if epsilon > 1 or epsilon < 0:
        raise ValueError("epsilon must be between 0 and 1.")
    if epsilon > 0 and random.random() < epsilon:
        return random.randrange(2)
    else:
        with torch.no_grad():
            q_values = model(observation)
            return q_values.argmax().item()

def update_q_batch(
        model, target_model, optimizer, transitions, gamma):
    batch_size = len(transitions)
    batch_inputs = torch.stack(
        [transition[0] for transition in transitions]
    )

    batch_actions = torch.tensor(
        [transition[1] for transition in transitions],
        dtype=torch.long,
    )

    batch_rewards = torch.tensor(
        [transition[2] for transition in transitions],
        dtype=torch.float64,
    )

    batch_next_inputs = torch.stack(
        [transition[3] for transition in transitions],
    )

    batch_terminated = torch.tensor(
        [transition[4] for transition in transitions],
        dtype=torch.bool,
    )

    with torch.no_grad():
        batch_next_q_values = target_model(batch_next_inputs)
        next_max_q_values = batch_next_q_values.max(dim=1).values

        batch_targets = torch.where(
            batch_terminated,
            batch_rewards,
            batch_rewards + gamma * next_max_q_values,
        )

    optimizer.zero_grad(set_to_none=True)

    batch_q_values = model(batch_inputs)
    batch_indices = torch.arange(batch_size)
    selected_q_values = batch_q_values[
        batch_indices,
        batch_actions,
    ]

    loss = (
        0.5 * (selected_q_values - batch_targets).square()
        ).mean()

    if not torch.isfinite(loss).item():
        raise RuntimeError("Training loss is not finite.")
    loss.backward()
    optimizer.step()

    for parameter in model.parameters():
        if not torch.isfinite(parameter).all().item():
            raise RuntimeError("Model parameters are not finite.")

    return loss.item()

def run_episode(
        model, optimizer, replay_buffer, epsilon,
        target_model, training=True, record_history=False):
    elapsed = 0.0
    height = 100.0
    velocity = -10.0
    propellant = 15.0
    dt = 0.02
    max_time = 30.0
    step_count = 0
    max_loss = None
    gamma = 1.0
    batch_size = 32

    history = []
    control_history = []

    if record_history:
        history.append((elapsed, height, velocity, propellant))

    inputs = torch.tensor(
        [
            height / 100,
            velocity / 50,
            propellant / 15,
            (max_time - elapsed) / max_time,
        ],
        dtype=torch.float64
    )

    while elapsed < max_time:
        action = select_action(model, inputs, epsilon=epsilon)
        commanded_thrust = 2000.0 if action == 1 else 0.0
        requested_dt = min(dt, max_time - elapsed)
        step_result = step_fueled_flight(
            height=height,
            velocity=velocity,
            dry_mass=100.0,
            propellant_mass=propellant,
            commanded_thrust=commanded_thrust,
            specific_impulse=200.0,
            g=9.81,
            dt=requested_dt,
        )
        ground_contact = step_result["ground_contact"]

        next_elapsed = min(
            max_time,
            elapsed + step_result["elapsed"]
        )

        time_limit_reached = (
            next_elapsed >= max_time and not ground_contact
        )
        terminated = ground_contact or time_limit_reached

        current_flight_status = classify_landing(
            velocity=step_result["velocity"],
            ground_contact=ground_contact,
            max_landing_speed=2)

        if current_flight_status == "landed":
            reward = 100.0
        elif current_flight_status == "crashed":
            reward = -min(abs(step_result["velocity"]), 100.0)
        elif time_limit_reached:
            reward = -100.0
        else:
            reward = 0.0

        next_inputs = torch.tensor(
            [
                step_result["height"] / 100,
                step_result["velocity"] / 50,
                step_result["propellant_mass"] / 15,
                (max_time - next_elapsed) / max_time,
            ],
            dtype=torch.float64,
        )
        if training:
            replay_buffer.append(
                (
                    inputs.clone(),
                    action,
                    reward,
                    next_inputs.clone(),
                    terminated,
                )
            )
            if len(replay_buffer) >= batch_size:
                transitions = random.sample(
                    list(replay_buffer),
                    batch_size,
                )
                loss_value = update_q_batch(
                    model, target_model, optimizer, transitions, gamma
                )

                if max_loss is None or loss_value > max_loss:
                    max_loss = loss_value

        if record_history:
            actual_thrust = (
                commanded_thrust if propellant > 0 else 0.0
            )
            control_history.append(
                (elapsed, next_elapsed, actual_thrust)
            )

        height = step_result["height"]
        velocity = step_result["velocity"]
        propellant = step_result["propellant_mass"]
        inputs = next_inputs
        elapsed = next_elapsed
        step_count += 1

        if record_history:
            history.append((elapsed, height, velocity, propellant))

        if terminated:
            break

    result = {
        "elapsed": elapsed,
        "step_count": step_count,
        "final_height": height,
        "final_velocity": velocity,
        "remaining_propellant": propellant,
        "status": current_flight_status,
        "termination_reason": (
            "ground_contact" if ground_contact else "time_limit"
        ),
        "final_reward": reward,
        "max_loss": max_loss,
    }

    if record_history:
        result["history"] = history
        result["control_history"] = control_history
    return result

if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)

    model = nn.Sequential(
        nn.Linear(4, 4, dtype=torch.float64),
        nn.ReLU(),
        nn.Linear(4, 2, dtype=torch.float64)
    )
    target_model = deepcopy(model)

    learning_rate = 0.001
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
    )

    replay_buffer = deque(maxlen=20_000)
    
    before_training = run_episode(
        model, optimizer, replay_buffer=replay_buffer,
        epsilon=0.0, target_model=target_model, training=False
    )
    print(f"Before training: {before_training}")

    best_reward = before_training["final_reward"]
    best_episode = 0
    best_state = deepcopy(model.state_dict())

    episode_count = 50
    epsilon_start = 0.8
    epsilon_end = 0.1
    evaluation_history = [(0, before_training)]
    for episode in range(1, episode_count + 1):
        progress = (episode - 1) / (episode_count - 1)
        epsilon = (
            epsilon_start
            + progress * (epsilon_end - epsilon_start)
        )

        result = run_episode(
            model, optimizer, replay_buffer=replay_buffer,
            epsilon=epsilon, target_model=target_model)
        target_model.load_state_dict(model.state_dict())

        print(f"--------------------\n"
              f"Episode {episode} | "
              f"Epsilon: {epsilon:.3f}\n"
              f"Result: {result}")

        if episode % 5 == 0 and episode != episode_count:
            mid_training = run_episode(
                model, optimizer, replay_buffer,
                epsilon=0.0, target_model=target_model, training=False
            )
            evaluation_history.append((episode, mid_training))

            if mid_training["final_reward"] > best_reward:
                best_reward = mid_training["final_reward"]
                best_episode = episode
                best_state = deepcopy(model.state_dict())

            print(f"--------------------\n"
                  f"Evaluation after episode {episode}: {mid_training}"
                  )
    after_training = run_episode(
        model, optimizer, replay_buffer=replay_buffer,
        epsilon=0.0, target_model=target_model, training=False
    )

    if after_training["final_reward"] > best_reward:
        best_reward = after_training["final_reward"]
        best_episode = episode_count
        best_state = deepcopy(model.state_dict())

    evaluation_history.append((episode_count, after_training))
    print(f"--------------------")
    print(f"After training: {after_training}")
    print(f"Stored transitions: {len(replay_buffer)}")


    output_dir = Path(__file__).resolve().parents[1] / "runs"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "evaluation_history.csv"
    fieldnames = ["episode"] + list(before_training.keys())

    with output_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for episode, evaluation in evaluation_history:
            row = evaluation.copy()
            row["episode"] = episode
            writer.writerow(row)

    print(f"--------------------")
    print(f"Evaluation history saved to {output_path}")

    model_path = output_dir / "final_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model weights saved to: {model_path}")

    best_model_path = output_dir / "best_model.pt"
    torch.save(best_state, best_model_path)

    print(
        f"Best evaluation: episode={best_episode}, "
        f"reward={best_reward:.3f}"
    )