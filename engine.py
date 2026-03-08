import torch
import random
from config import config


def train_epoch(model, loss_fn, optimiser, train_loader, config):
    k = config['k']
    n_trajectory = config['n_trajectory'] - 1
    model.train()
    epoch_loss = 0
    for (X_batch,) in train_loader:
        optimiser.zero_grad()
        t = random.randint(0, n_trajectory - k - 1)
        x0 = X_batch[:, t, :]       # (8, 3) — starting points
        target = X_batch[:, t+k, :] # (8, 3) — k steps ahead

        x = x0
        for _ in range(k):
            x = model(x)
        loss = loss_fn(x, target)
        loss.backward()
        optimiser.step()
        epoch_loss += loss.item()
    return epoch_loss / len(train_loader)


def val_epoch(model, loss_fn, val_loader, config):
    k = config['k']
    n_trajectory = config['n_trajectory'] - 1
    model.eval()
    val_loss = 0
    with torch.inference_mode():
        for (X_batch,) in val_loader:
            t = random.randint(0, n_trajectory - k - 1)
            x0 = X_batch[:, t, :]       # (8, 3) — starting points
            target = X_batch[:, t+k, :] # (8, 3) — k steps ahead

            x = x0
        for _ in range(k):
            x = model(x)
        val_loss += loss_fn(x, target).item()
    return val_loss / len(val_loader)


def train(model, config, loss_fn, optimiser, train_loader, val_loader):
    history = {'train': [], 'val': []}
    n_epochs = config['n_epochs']

    for epoch in range(n_epochs):
        train_loss = train_epoch(model, loss_fn, optimiser, train_loader, config)
        val_loss   = val_epoch(model, loss_fn, val_loader, config)
        history['train'].append(train_loss)
        history['val'].append(val_loss)

        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}/{n_epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f}')

    return history


def test(model, loss_fn, test_loader, config):
    model.eval()
    test_loss = 0
    preds_all = []
    k = config['k']
    n_trajectory = config['n_trajectory'] - 1
    with torch.inference_mode():
        for (X_batch,) in test_loader:
            t = random.randint(0, n_trajectory - k - 1)
            x0     = X_batch[:, t, :]
            target = X_batch[:, t+k, :]
            x = x0
            for _ in range(k):
                x = model(x)
            test_loss += loss_fn(x, target).item()
            preds_all.append(x.cpu())
    test_loss /= len(test_loader)
    preds_all = torch.cat(preds_all, dim=0)
    return preds_all, test_loss