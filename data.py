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

    n_total = n_samples * (n_trajectory - 1)
    data = np.zeros((n_total, 3))
    labels = np.zeros((n_total, 3))

    idx = 0
    for _ in range(n_samples):
        x0 = np.array([
            np.random.uniform(-15, 15),
            np.random.uniform(-15, 15),
            np.random.uniform(0, 40)
        ])
        for _ in range(n_trajectory - 1):
            data[idx] = x0
            x0 = rk4(lorentz, h, x0, sigma, beta, rho)
            labels[idx] = x0
            idx += 1

    return data, labels


def get_loaders(config):
    data, labels = generate_data(config)

    # Split
    n = len(data)
    train_end = int(n * config['train_frac'])
    val_end   = int(n * config['val_frac'])

    train_data,  train_labels = data[:train_end],         labels[:train_end]
    val_data,    val_labels   = data[train_end:val_end],  labels[train_end:val_end]
    test_data,   test_labels  = data[val_end:],           labels[val_end:]

    # Tensors
    to_tensor = lambda x: torch.tensor(x, dtype=torch.float32)
    train_data,  train_labels = to_tensor(train_data),  to_tensor(train_labels)
    val_data,    val_labels   = to_tensor(val_data),    to_tensor(val_labels)
    test_data,   test_labels  = to_tensor(test_data),   to_tensor(test_labels)

    # Normalise
    mean = train_data.mean(dim=0)
    std  = train_data.std(dim=0) + 1e-8

    train_data,  train_labels = (train_data - mean) / std,  (train_labels - mean) / std
    val_data,    val_labels   = (val_data   - mean) / std,  (val_labels   - mean) / std
    test_data,   test_labels  = (test_data  - mean) / std,  (test_labels  - mean) / std

    # Device
    device = config['device']
    train_data,  train_labels = train_data.to(device),  train_labels.to(device)
    val_data,    val_labels   = val_data.to(device),    val_labels.to(device)
    test_data,   test_labels  = test_data.to(device),   test_labels.to(device)

    # Loaders
    train_loader = DataLoader(TensorDataset(train_data,  train_labels),  batch_size=config['batch_size'], shuffle=True)
    val_loader   = DataLoader(TensorDataset(val_data,    val_labels),    batch_size=config['batch_size'], shuffle=False)
    test_loader  = DataLoader(TensorDataset(test_data,   test_labels),   batch_size=config['batch_size'], shuffle=False)

    return train_loader, val_loader, test_loader, mean, std