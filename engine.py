import torch
from torchdiffeq import odeint

def train_epoch(model, loss_fn, optimiser, train_loader, device):
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        t_span = torch.tensor([0.0, 0.01]).to(device)
        preds = odeint(model, X_batch, t_span)[-1]
        loss = loss_fn(preds, y_batch)
        loss.backward()
        optimiser.step()
        epoch_loss += loss.item()
    return epoch_loss / len(train_loader)


def val_epoch(model, loss_fn, val_loader, device):
    model.eval()
    val_loss = 0
    with torch.inference_mode():
        for X_batch, y_batch in val_loader:
            t_span = torch.tensor([0.0, 0.01]).to(device)
            preds = odeint(model, X_batch, t_span)[-1]
            val_loss += loss_fn(preds, y_batch).item()
    return val_loss / len(val_loader)


def train(model, config, loss_fn, optimiser, train_loader, val_loader):
    history = {'train': [], 'val': []}
    n_epochs = config['n_epochs']

    for epoch in range(n_epochs):
        train_loss = train_epoch(model, loss_fn, optimiser, train_loader, config['device'])
        val_loss   = val_epoch(model, loss_fn, val_loader, config['device'])
        history['train'].append(train_loss)
        history['val'].append(val_loss)

        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}/{n_epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f}')

    return history


def test(model, loss_fn, test_loader, device):
    model.eval()
    test_loss = 0
    preds_all = []
    with torch.inference_mode():
        for X_batch, y_batch in test_loader:
            t_span = torch.tensor([0.0, 0.01]).to(device)
            preds = odeint(model, X_batch, t_span)[-1]
            test_loss += loss_fn(preds, y_batch).item()
            preds_all.append(preds.cpu())
    test_loss /= len(test_loader)
    preds_all = torch.cat(preds_all, dim=0)
    return preds_all, test_loss