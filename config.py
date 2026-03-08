import torch

config = {
    # System
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',

    # Lorenz parameters
    'sigma': 10,
    'rho': 28,
    'beta': 8/3,
    

    # Data generation
    'n_samples': 100,
    'n_trajectory': 3000,
    'h': 0.01,
    

    # Train/val/test split
    'train_frac': 0.8,
    'val_frac': 0.9,

    # Training
    'n_epochs': 50,
    'batch_size': 256,
    'lr': 1e-3,

    # Eval
    'n_rollout_steps': 2999,
    'ic1': [0.1, 1.0, 0.0],
    'ic2': [0.2, 1.0, 0.0],
}