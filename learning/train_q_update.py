import torch
from torch import nn

if __name__ == "__main__":
    torch.manual_seed(0)

    inputs = torch.tensor(
        [0.2, -0.4, 0.5],
        dtype=torch.float64,
    )
    next_inputs = torch.tensor(
        [0.18, -0.35, 0.49],
        dtype=torch.float64,
    )
    model = nn.Sequential(
        nn.Linear(3, 4, dtype=torch.float64),
        nn.ReLU(),
        nn.Linear(4, 2, dtype=torch.float64)
    )

    learning_rate = 0.1
    action = 1
    reward = -0.1
    gamma = 0.9
    terminated = False

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
    )


    for step in range(1, 2):
        with torch.no_grad():
            if terminated:
                target = torch.tensor(reward, dtype=torch.float64)
            else:
                next_q_values = model(next_inputs)
                target = reward + gamma * next_q_values.max()
        optimizer.zero_grad(set_to_none=True)

        q_values = model(inputs)
        selected_q = q_values[action]
        loss = 0.5 * (selected_q - target).square().sum()

        loss.backward()
        optimizer.step()
        with torch.no_grad():
            q_updated = model(inputs)

        print(
            f"Q before update: {q_values[action]:.6f}\n"
            f"Target for this update: {target.item():.6f}\n"
            f"Q after update = {q_updated[action]:.6f}"
        )