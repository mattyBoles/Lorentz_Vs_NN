from torch import nn
import numpy

class chaos_model(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(in_features=3, out_features=64)
    self.relu = nn.ReLU()
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