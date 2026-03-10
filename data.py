import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from lorentz import rk4, lorentz

def generate_data(config):
    sigma = config['sigma']
    rho = config['rho']
    beta = config['beta']
    n_samples = config['n_samples']
    n_trajectory = config['n_trajectory']
    h = config['h']

    data = np.zeros((n_samples,n_trajectory - 1, 3))

    idx = 0
    for _ in range(n_samples):
        idy = 0
        x0 = np.array([
            np.random.uniform(-15, 15),
            np.random.uniform(-15, 15),
            np.random.uniform(0, 40)
        ])
        for _ in range(n_trajectory - 1):
            data[idx, idy] = x0
            x0 = rk4(lorentz, h, x0, sigma, beta, rho)
            idy += 1
        idx += 1

    return data


def get_loaders(config):
    data = generate_data(config)

    # Split
    n = len(data)
    train_end = int(n * config['train_frac'])
    val_end   = int(n * config['val_frac'])

    train_data = data[:train_end]         
    val_data   = data[train_end:val_end]
    test_data  = data[val_end:]

    # Tensors
    to_tensor = lambda x: torch.tensor(x, dtype=torch.float32)
    train_data = to_tensor(train_data)
    val_data = to_tensor(val_data)
    test_data = to_tensor(test_data)

    # Normalise
    mean = train_data.mean(dim=(0,1))
    std  = train_data.std(dim=(0,1)) + 1e-8

    train_data = (train_data - mean) / std
    val_data   = (val_data   - mean) / std
    test_data  = (test_data  - mean) / std

    # Device
    device = config['device']
    train_data = train_data.to(device)
    val_data   = val_data.to(device)
    test_data  = test_data.to(device)

    # Loaders
    train_loader = DataLoader(TensorDataset(train_data),  batch_size=config['batch_size'], shuffle=True)
    val_loader   = DataLoader(TensorDataset(val_data),    batch_size=config['batch_size'], shuffle=False)
    test_loader  = DataLoader(TensorDataset(test_data),   batch_size=config['batch_size'], shuffle=False)

    return train_loader, val_loader, test_loader, mean, std