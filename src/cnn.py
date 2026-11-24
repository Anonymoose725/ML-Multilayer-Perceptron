"""PyTorch CNN and its training loop."""
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module): # configurable size
  """conv -> ReLU -> pool -> conv -> ReLU -> pool -> FC(256) -> ReLU -> FC(10).  Output = logits"""
  def __init__(self, in_ch=1, img=28, c1=32, c2=64, k=3, hidden=256, n_classes=10):
    super().__init__()
    self.conv1 = nn.Conv2d(in_ch, c1, k, padding=k // 2) # first conv layer
    self.conv2 = nn.Conv2d(c1, c2, k, padding=k // 2) # second conv layer
    self.pool = nn.MaxPool2d(2) # 2x2 max pooling halves height and width
    self.fc1 = nn.Linear(c2 * (img // 4) ** 2, hidden) # FC hidden layer
    self.fc2 = nn.Linear(hidden, n_classes) # FC output layer (one logit per class)

  def forward(self, x): # forward pass
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    return self.fc2(F.relu(self.fc1(x.flatten(1))))  # flatten, FC1 to ReLU, FC2 to logits (no softmax cause loss adds it)
