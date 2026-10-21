import torch


if __name__ == "__main__":
    inputs = torch.tensor(
        [0.2, -0.4, 0.5],
        dtype=torch.float64,
    )
    weights = torch.tensor(
        [1.0, -0.5, 0.2],
        dtype=torch.float64,
        requires_grad=True,
    )
    bias = torch.tensor(
        -0.3,
        dtype=torch.float64,
        requires_grad=True,
    )
    target = torch.tensor(-1.0, dtype=torch.float64)
    learning_rate = 0.1
    optimizer = torch.optim.SGD(
        [weights, bias],
        lr=learning_rate,
    )

    prediction = (inputs * weights).sum() + bias
    loss = 0.5 * (prediction - target) ** 2
    print(
        f"Step: INIT | "
        f"Prediction: {prediction.item():.6f} | "
        f"Loss: {loss.item():.3e}"
    )

    for step in range(1, 101):
        optimizer.zero_grad(set_to_none=True)
        prediction = (inputs * weights).sum() + bias
        loss = 0.5 * (prediction - target) ** 2
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            updated_prediction = (inputs * weights).sum() + bias
            updated_loss = 0.5 * (updated_prediction - target) ** 2
        if step == 1 or step % 10 == 0:
            print(
                f"Step: {step:3d} | "
                f"Prediction: {updated_prediction.item():.6f} | "
                f"Loss: {updated_loss.item():.3e}"
            )