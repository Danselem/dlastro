import torch

x = torch.randn(10000, 10000, device="cuda")
y = x @ x.T

torch.cuda.synchronize()

print("Device:", y.device)
print("Result shape:", y.shape)
print("Allocated VRAM:", round(torch.cuda.memory_allocated() / 1024**3, 2), "GB")