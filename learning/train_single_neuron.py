def neuron_output(inputs, weights, bias):
    if len(inputs) != len(weights):
        raise ValueError("The number of inputs must be same with that of weights.")
    z = bias
    for i in range(len(inputs)):
        z += inputs[i] * weights[i]

    return z

def squared_error(prediction, target):
    return 0.5 * (prediction - target) ** 2


if __name__ == "__main__":
    inputs = [0.2, -0.4, 0.5]
    weights = [1.0, -0.5, 0.2]
    bias = -0.3
    target = -1.0
    learning_rate = 0.1

    prediction = neuron_output(inputs, weights, bias)
    loss = squared_error(prediction, target)
    print(
        f"Step: INIT | "
        f"Prediction: {prediction:.6f} | "
        f"Loss: {loss:.6f}"
    )
    for step in range(1, 101):
        gradient_z = prediction - target
        for i in range(len(weights)):
            weights[i] -= learning_rate * gradient_z * inputs[i]
        bias -= learning_rate * gradient_z

        updated_prediction = neuron_output(inputs, weights, bias)
        updated_loss = squared_error(updated_prediction, target)

        if step == 1 or step % 10 == 0:
            print(
                f"Step: {step:3d} | "
                f"Prediction: {updated_prediction:.6f} | "
                f"Loss: {updated_loss:.6f}"
            )

        prediction = updated_prediction
        loss = updated_loss