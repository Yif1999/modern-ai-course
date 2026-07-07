import mlx.core as mx

print("=== MLX Autograd Demo ===")

x = mx.array([1.0, 2.0, 3.0])
w = mx.array([0.1, 0.2, 0.3])


def f(w):
    return mx.sum(x * w)


y, grad_w = mx.value_and_grad(f)(w)

mx.eval(y, grad_w)

print("x =", x)
print("w =", w)
print("y =", y)
print("grad_w =", grad_w)
