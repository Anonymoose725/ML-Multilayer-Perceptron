"""Dataset loading, vectorising, normalization, augmentation."""
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset
from src.config import SEED
from src.config import VAL_SPLIT
from src.config import WORKERS


class Vectorise:
  """Converts PIL image from FashionMNIST -> flat float32 vector"""
  def __init__(self, mean=None, std=None, rescale=True):
    self.mean, self.std, self.rescale = mean, std, rescale

  def __call__(self, img):
    x = np.array(img, dtype=np.float32).reshape(-1) # 784 values in [0, 255]
    if self.rescale:
      x /= 255. # sends to [0, 1]
    if self.mean is not None and self.std is not None:
      x = (x - self.mean) / self.std
    return x


def to_arrays(dataset, batch_size=1024):
  """Converts a dataset of (vector, label) pairs into numpy arrays X (N, 784), y (N,)"""
  dl = DataLoader(dataset, batch_size=batch_size, shuffle=False)
  Xs, ys = [], []

  for xb, yb in dl:
    Xs.append(xb.numpy())
    ys.append(yb.numpy())

  return np.concatenate(Xs).astype(np.float32), np.concatenate(ys).astype(np.int64)


# -- augmentations
def build_augmentations():
  """
  Returns a list of augmentation transforms to apply to training images
  """
  return [transforms.RandomHorizontalFlip(), transforms.ColorJitter(brightness=0.5, contrast=0.5)]



def make_split(n, seed=SEED, val_split=VAL_SPLIT): # was globals
    perm = np.random.default_rng(seed).permutation(n)
    n_val = int(val_split * n)
    return perm[n_val:], perm[:n_val] # train_idx, val_idx

# call to obtain typical numpy appropriate data!
def load_data(normalize=True, mean=None, std=None, train_idx=None, val_idx=None):
  """
  Returns X_train, y_train, X_val, y_val, X_test, y_test as numpy arrays
  Inputs:
  normalize:  a boolean flag to normalize the loaded dataset, true by default

  Output:
  X_train, y_train, X_val, y_val, T_test, y_test:   numpy arrays converted from vectorised (vector, label) pairs into (N, 784), y(N,)
  """
  
  tf = Vectorise(mean, std) if normalize else Vectorise(rescale=False) # unnormalized = raw 0-255 (needed for Q4)

  train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=tf)
  test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=tf) # again, dont augment the test data (obviously!)

  return (*to_arrays(Subset(train_data, train_idx)),
          *to_arrays(Subset(train_data, val_idx)),
          *to_arrays(test_data))


def build_sets(ds_cls, tf_eval, tf_aug=None, val_frac=VAL_SPLIT):
  """
  Same split as the MLP section (default_rng(SEED).permutation), so FashionMNIST reproduces train_idx/val_idx.
  Returns clean train, clean val, test, augmented train (None if tf_aug is None), train_idx, val_idx

  Most useful in Q9
  """
  full_eval = ds_cls('./data', train=True, download=True, transform=tf_eval) # whole training file with clean transform
  test = ds_cls('./data', train=False, download=True, transform=tf_eval)
  t_idx, v_idx = make_split(len(full_eval), val_split=val_frac)
  aug = None # default
  if tf_aug is not None:  # only build it when an augmentation pipeline is given
    aug = Subset(ds_cls('./data', train=True, download=True, transform=tf_aug), t_idx)  # augmented copy, training indices only
  return Subset(full_eval, t_idx), Subset(full_eval, v_idx), test, aug, t_idx, v_idx  # validation is never augmented


def cache_tensors(ds, bs=1000):
  """
  Run transform pipeline once and keep the tensors, so clean sets aren't re-decoded each epoch

  Helps speed up runs.
  """
  xs, ys = [], []
  for xb, yb in DataLoader(ds, batch_size=bs, shuffle=False, num_workers=WORKERS):
    xs.append(xb); ys.append(yb)
  return TensorDataset(torch.cat(xs), torch.cat(ys)) # glue batches together into 1 dataset
