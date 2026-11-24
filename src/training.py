"""PyTorch training/evaluation loop shared by the CNN and ResNet experiments."""
import torch
from torch.utils.data import DataLoader, Subset
import time
import torch.nn.functional as F
from src.config import BATCH
from src.config import device


def sync():
  if device.type == 'cuda': torch.cuda.synchronize()
  elif device.type == 'mps': torch.mps.synchronize()


def loader(ds, shuffle=False, bs=BATCH, workers=0): # shorthand for creating DataLoader
  return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=workers)


def evaluate_torch(model, dl): # mean loss and accuracy of a model over a whole loader
  model.eval()
  loss_sum, correct, n = 0.0, 0, 0
  with torch.no_grad():
    for xb, yb in dl:
      xb, yb = xb.to(device), yb.to(device)
      out = model(xb)
      loss_sum += F.cross_entropy(out, yb, reduction='sum').item()
      correct += (out.argmax(1) == yb).sum().item()
      n += len(yb)
  return loss_sum / n, correct / n


def train_torch(model, train_dl, eval_dls, epochs, lr=0.05, momentum=0.9, weight_decay=0.0, eval_model=None, logging=True):
  model.to(device)
  eval_model = model if eval_model is None else eval_model.to(device)
  params = [p for p in model.parameters() if p.requires_grad]  # frozen layers aren't updated
  opt = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
  stats = {k: [] for k in ('train_loss', 'train_acc', 'val_loss', 'val_acc', 'test_loss', 'test_acc', 'elapsed')}
  total = 0.0
  for epoch in range(epochs):
    model.train()
    sync(); t0 = time.perf_counter()
    for xb, yb in train_dl:
      xb, yb = xb.to(device), yb.to(device)
      opt.zero_grad()
      loss = F.cross_entropy(model(xb), yb)  # softmax is inside the cross-entropy
      loss.backward()
      opt.step()
    sync(); total += time.perf_counter() - t0
    stats['elapsed'].append(total)
    for name, dl in eval_dls.items():
      l, a = evaluate_torch(eval_model, dl)
      stats[f'{name}_loss'].append(l); stats[f'{name}_acc'].append(a)
    if logging and (epoch + 1) % 5 == 0:
      print(f"Epoch #{epoch+1}/{epochs}: train acc {stats['train_acc'][-1]:.4f} | val acc {stats['val_acc'][-1]:.4f} | test acc {stats['test_acc'][-1]:.4f}")
  return stats
