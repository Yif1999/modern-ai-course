import torch

print("=== PyTorch Autograd Demo ===")

x = torch.tensor([1.0, 2.0, 3.0])
w = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)

y = (x * w).sum()

print("x =", x)
print("w =", w)
print("y =", y)

y.backward()

print("w.grad =", w.grad)
