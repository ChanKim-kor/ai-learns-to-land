import torch
from torch import nn

if __name__ == "__main__":
    torch.manual_seed(0)

    inputs = torch.tensor(
        [0.2, -0.4, 0.5],
        dtype=torch.float64,
    )
    model = nn.Sequential(
        nn.Linear(3, 4, dtype=torch.float64),
        nn.ReLU(),
        nn.Linear(4, 2, dtype=torch.float64)
    )
    target = torch.tensor([-1.0], dtype=torch.float64)
    learning_rate = 0.1
    action = 1

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
    )

    q_values = model(inputs)
    selected_q = q_values[action]
    loss = 0.5 * (selected_q - target).square().sum()

    print(
        f"Step: INIT | "
        f"Q values: {q_values.tolist()} | "
        f"Loss: {loss.item():.3e}"
    )

    for step in range(1, 101):
        optimizer.zero_grad(set_to_none=True)
        q_values = model(inputs)
        selected_q = q_values[action]
        loss = 0.5 * (selected_q - target).square().sum()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            updated_q_values = model(inputs)
            updated_loss = 0.5 * (updated_q_values[action] - target).square().sum()
        if step == 1 or step % 10 == 0:
            print(
                f"Step: {step:3d} | "
                f"Q values: {updated_q_values.tolist()} | "
                f"Loss: {updated_loss.item():.3e}"
            )