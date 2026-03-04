import torch
import numpy as np
from lorentz import rk4, lorentz
from torch.utils.data import TensorDataset, DataLoader
from models import chaos_model
from torch.optim.lr_scheduler import LinearLR, ConstantLR, SequentialLR
import matplotlib.pyplot as plt


def train(model, n_epochs, loss_fn, optimiser, train_loader, val_loader):
    history = {'train': [], 'val': []}

    for epoch in range(n_epochs):
        train_loss = train_epoch(model, loss_fn, optimiser, train_loader)
        val_loss = val_epoch(model, loss_fn, val_loader)
    
        history['train'].append(train_loss)
        history['val'].append(val_loss)
        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        return history

def train_epoch(model, loss_fn, optimiser, train_loader):
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        optimiser.zero_grad()
        preds = model(X_batch)
        loss = loss_fn(preds, y_batch)
        loss.backward()
        optimiser.step()
        epoch_loss += loss.item()

    epoch_loss /= len(train_loader)

    return epoch_loss

def val_epoch(model, loss_fn, val_loader):
    model.eval()
    val_loss = 0
    with torch.inference_mode():
        for X_batch, y_batch in val_loader:
            preds = model(X_batch)
            val_loss += loss_fn(preds, y_batch).item()
    val_loss /= len(val_loader)

    return val_loss

def test(model, loss_fn, test_loader):
    model.eval()
    test_loss = 0
    preds_all =[]
    with torch.inferance_mode():
        for X_batch, y_batch in test_loader:
            preds = model(X_batch)
            test_loss += loss_fn(preds, y_batch).item()
            preds_all.append(preds.cpu())
    test_loss /= len(test_loader)
    preds_all = torch.cat(preds_all, dim=0)
    return(preds_all, test_loss)