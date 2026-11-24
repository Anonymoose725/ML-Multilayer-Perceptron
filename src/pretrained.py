"""Frozen ResNet-18 + trainable FC head (transfer learning)."""
import torch
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
from torch.utils.data import TensorDataset
from torchvision import models
from src.config import WORKERS
from src.config import device


def make_backbone(): # frozen ResNet feature extractor
  m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT) # load ImageNet pre-trained weights (downloads on first use)
  m.fc = nn.Identity()
  for p in m.parameters():
    p.requires_grad = False # freeze it
  return m.to(device).eval() # move to device; eval mode so BatchNorm uses its stored statistics


@torch.no_grad() # no gradients needed while extracting features
def extract_features(backbone, ds, bs=500): # run images through the frozen backbone once
  backbone.eval() # make sure BatchNorm is frozen
  fs, ys = [], [] # feature batches and labels
  for xb, yb in DataLoader(ds, batch_size=bs, shuffle=False, num_workers=WORKERS):
    fs.append(backbone(xb.to(device)).cpu()); ys.append(yb)
  return TensorDataset(torch.cat(fs), torch.cat(ys)) # features + labels as a dataset


def make_head(n_hidden, width=256, in_dim=512, n_classes=10): # trainable fully connected classifier
  layers, d = [], in_dim # list of layers; d is current feature size
  for _ in range(n_hidden): # one block per hidden layer
    layers += [nn.Linear(d, width), nn.ReLU()]; d = width # Linear to ReLU
  layers.append(nn.Linear(d, n_classes)) # logits
  return nn.Sequential(*layers)


class Pretrained(nn.Module):
  """Frozen backbone + trainable head. train() keeps the backbone in eval mode"""
  def __init__(self, backbone, head): # takes the two parts
    super().__init__()
    self.backbone, self.head = backbone, head

  def train(self, mode=True): # model.train() must not un-freeze BatchNorm
    super().train(mode)
    self.backbone.eval()
    return self

  def forward(self, x): # forward pass
    with torch.no_grad():
      f = self.backbone(x).flatten(1)
    return self.head(f)
