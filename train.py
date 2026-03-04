import torch
import numpy as np
from lorentz import rk4, lorentz
from torch.utils.data import TensorDataset, DataLoader
from models import chaos_model
from torch.optim.lr_scheduler import LinearLR, ConstantLR, SequentialLR
import matplotlib.pyplot as plt
from engine import train, test

np.random.seed(42)
torch.manual_seed(42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

params = {'sigma': 10, 'rho': 28, 'beta': 8/3}

n_samples = 100
n_trajectory = 3000
h = 0.01

n_total = n_samples * (n_trajectory - 1)
train_data = np.zeros((n_total, 3))
train_labels = np.zeros((n_total, 3))

idx = 0
for _ in range(n_samples):
  x0_ = np.array([
    np.random.uniform(-15,15),
    np.random.uniform(-15,15),
    np.random.uniform(0,40)
    ])


  for _ in range(n_trajectory - 1):
    train_data[idx] = x0_
    x_ = rk4(lorentz, h, x0_, params['sigma'], params['beta'], params['rho'])
    train_labels[idx] = x_
    x0_ = x_
    idx += 1

n = len(train_data)
train_end = int(n * 0.8)
val_end = int(n * 0.9)

val_data = train_data[train_end:val_end]
val_labels = train_labels[train_end:val_end]
test_data = train_data[val_end:]
test_labels = train_labels[val_end:]
train_data = train_data[:train_end]
train_labels = train_labels[:train_end]

train_data   = torch.tensor(train_data,   dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.float32)
val_data     = torch.tensor(val_data,     dtype=torch.float32)
val_labels   = torch.tensor(val_labels,   dtype=torch.float32)
test_data    = torch.tensor(test_data,    dtype=torch.float32)
test_labels  = torch.tensor(test_labels,  dtype=torch.float32)

train_mean = train_data.mean(dim=0)
train_std = train_data.std(dim=0)

train_data =  (train_data - train_mean)/(train_std + 1e-8)
train_labels =  (train_labels - train_mean)/(train_std + 1e-8)
test_data =  (test_data - train_mean)/(train_std + 1e-8)
test_labels =  (test_labels - train_mean)/(train_std + 1e-8)

train_data   = train_data.to(device)
train_labels = train_labels.to(device)
val_data     = val_data.to(device)
val_labels   = val_labels.to(device)
test_data    = test_data.to(device)
test_labels  = test_labels.to(device)


train_dataset = TensorDataset(train_data, train_labels)
test_dataset = TensorDataset(test_data, test_labels)
val_dataset = TensorDataset(val_data, val_labels)

train_loader = DataLoader(train_dataset, batch_size = 256, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = 256, shuffle = False)
val_loader = DataLoader(val_dataset, batch_size = 256, shuffle = False)


model = chaos_model().to(device)

print(model)
print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')


loss_fn = torch.nn.MSELoss()
optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)

n_epochs = 50

history = train(model=model, n_epochs=n_epochs, loss_fn = loss_fn, optimiser=optimiser, train_loader=train_loader, val_loader=val_loader)

preds, test_loss = test(model, loss_fn, test_loader)

print(f'Test Loss: {test_loss:.6f}')
print(preds[:3], test_labels[:3])