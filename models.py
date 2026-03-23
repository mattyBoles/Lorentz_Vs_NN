import torch.nn as nn
import numpy
from torchdiffeq import odeint

class chaos_model(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(in_features=3, out_features=64)
    self.relu = nn.Tanh()
    self.dropout = nn.Dropout(0)
    self.fc2 = nn.Linear(in_features =64, out_features = 64)
    self.fc3 = nn.Linear(in_features=64, out_features = 3)
  def forward(self, x):
    x = self.fc1(x)
    x = self.relu(x)
    x = self.dropout(x)

    x = self.fc2(x)
    x = self.relu(x)
    x = self.dropout(x)

    x = self.fc3(x)

    return x


class LorenzODE(nn.Module):
  def __init__(self):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(3, 64),
      nn.Tanh(),
      nn.Linear(64, 64),
      nn.Tanh(),
      nn.Linear(64, 3)
    )

  def forward(self, t, x):
    return self.net(x)

