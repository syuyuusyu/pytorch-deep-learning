import torch;

x = torch.ones((1024 * 12, 1024 * 12), dtype=torch.float32,device='mps')

print(x)