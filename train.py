import torch
import numpy as np
from config import config
from data import get_loaders
from models import chaos_model
from engine import train, test
from eval import evaluate

# --- Reproducibility ---
np.random.seed(42)
torch.manual_seed(42)

# --- Data ---
train_loader, val_loader, test_loader, mean, std = get_loaders(config)

# --- Model ---
model = chaos_model().to(config['device'])
print(model)
print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')

# --- Train ---
loss_fn   = torch.nn.MSELoss()
optimiser = torch.optim.AdamW(model.parameters(), lr=config['lr'])
history   = train(model, config, loss_fn, optimiser, train_loader, val_loader)

# --- Test ---
preds, test_loss = test(model, loss_fn, test_loader, config)
print(f'Test Loss: {test_loss:.6f}')

# --- Evaluate ---
evaluate(model, history, config, mean, std)

# torch.save(model.state_dict(), 'model.pth')