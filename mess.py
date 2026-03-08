from models import chaos_model
import torch
from eval import evaluate
from config import config

model = chaos_model()
model.load_state_dict(torch.load('model.pth', map_location='cpu'))
model.eval()

mean = torch.tensor([-0.1088, -0.1050, 23.8636])
std = torch.tensor([7.9852, 8.9939, 8.3128])
history = None

evaluate(model, history, config, mean, std)
