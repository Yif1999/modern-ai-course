import mlx.core as mx
import mlx.optimizers as optim

print("=== MLX Linear Regression with Optimizer ===")

x = mx.array([1.0, 2.0, 3.0, 4.0])
y_true = 3 * x + 2

params = {
    "w": mx.array(0.0),
    "b": mx.array(0.0),
}

lr = 0.01
steps = 200

optimizer = optim.SGD(learning_rate=lr)


def loss_fn(params):
    y_pred = params["w"] * x + params["b"]
    loss = mx.mean((y_pred - y_true) ** 2)
    return loss


value_and_grad_fn = mx.value_and_grad(loss_fn)

for step in range(steps):
    loss, grads = value_and_grad_fn(params)

    params = optimizer.apply_gradients(grads, params)

    mx.eval(loss, params, optimizer.state)

    if step % 20 == 0 or step == steps - 1:
        print(
            f"step={step:03d} "
            f"loss={float(loss):.6f} "
            f"w={float(params['w']):.4f} "
            f"b={float(params['b']):.4f}"
        )

print("Expected: w ≈ 3, b ≈ 2")
