# Linear Regression Notes

## Goal

We train a tiny linear model:

```text
y_pred = w * x + b
```

The true rule is:

```text
y = 3x + 2
```

So after training, we expect:

```text
w ≈ 3
b ≈ 2
```

## Loss

We use mean squared error:

```text
loss = mean((y_pred - y_true)^2)
```

Loss measures how wrong the prediction is.

## Gradient Descent

The update rule is:

```text
parameter = parameter - learning_rate * gradient
```

If the gradient points toward increasing loss, subtracting it moves the parameter toward lower loss.

## PyTorch Style

- Use `requires_grad=True` on trainable parameters.
- Compute loss.
- Call `loss.backward()`.
- Read gradients from `w.grad` and `b.grad`.
- Update parameters inside `torch.no_grad()`.
- Clear gradients after every step.

## MLX Style

- Define a loss function.
- Use `mx.value_and_grad(loss_fn)(params)`.
- It returns both loss and gradients.
- Update params manually.
- Use `mx.eval(...)` to force lazy computations to run.

## Key Difference

PyTorch style:

```text
loss.backward() -> gradients are stored on parameters
```

MLX style:

```text
value_and_grad(loss_fn) -> gradients are returned from a function
```

---

# Optimizer Notes

## Why Optimizer Exists

In the previous lesson, we manually updated parameters:

```text
parameter = parameter - learning_rate * gradient
```

An optimizer automates this update rule.

## PyTorch Optimizer

```python
optimizer = torch.optim.SGD([w, b], lr=lr)
```

The optimizer receives the trainable parameters.

A standard PyTorch training step is:

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

Meaning:

1. Clear old gradients.
2. Compute new gradients.
3. Update parameters.

## MLX Optimizer

In MLX, parameters are often stored in dictionaries or modules.

A standard MLX optimizer step is:

```python
loss, grads = mx.value_and_grad(loss_fn)(params)
params = optimizer.apply_gradients(grads, params)
mx.eval(loss, params, optimizer.state)
```

Meaning:

1. Compute loss and gradients from a function.
2. Apply gradients to parameters.
3. Force lazy computations to run.

## Key Idea

SGD still means:

```text
parameter = parameter - learning_rate * gradient
```

The optimizer does not change the basic idea. It only packages the update logic.

Later, Adam and AdamW will use more advanced update rules, but the training loop structure stays the same.
