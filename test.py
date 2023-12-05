import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_circles
from icecream import ic

x= torch.tensor([1,2,3,4])
y = torch.tensor([1,2,3,5])

a = x.eq(y).sum().item()

print(a)

best_val_loss = float('inf')
print(best_val_loss)
