import mlx.core as mx

print("=== MLX Linear Regression Demo ===")

# 训练数据：真实规律是 y = 3x + 2
x = mx.array([1.0, 2.0, 3.0, 4.0])
y_true = 3 * x + 2

# 参数用一个数组表示：[w, b]
params = mx.array([0.0, 0.0])

lr = 0.01
steps = 200


def loss_fn(params):
    w = params[0]
    b = params[1]
    y_pred = w * x + b
    loss = mx.mean((y_pred - y_true) ** 2)
    return loss


value_and_grad_fn = mx.value_and_grad(loss_fn)

for step in range(steps):
    loss, grads = value_and_grad_fn(params)

    # manual gradient descent
    params = params - lr * grads

    mx.eval(loss, grads, params)

    if step % 20 == 0 or step == steps - 1:
        print(
            f"step={step:03d} "
            f"loss={float(loss):.6f} "
            f"w={float(params[0]):.4f} "
            f"b={float(params[1]):.4f}"
        )

print("Expected: w ≈ 3, b ≈ 2")
