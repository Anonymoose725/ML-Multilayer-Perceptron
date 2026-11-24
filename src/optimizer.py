"""Minibatch gradient descent with momentum and gradient clipping."""
import numpy as np
from src.mlp import MLP


class GradientDescent:

  def __init__(self, lr=0.01, batch_size=128, nb_epochs=50, clip_threshold=None, momentum=0.0):
    """
    Minibatch stochastic gradient descent. Holds hyperparameters and update method,
    while MLP.fit runs epoch-on-batch loop and calls run() once per batch

    Inputs:
    lr:             learning rate / step size
    batch_size:     number of samples in a minibatch (1 = pure stochastic, N = full batch GD)
    nb_epochs:      number of training set passes
    clip_threshold: max size tolerated until rescaling gradients, to prevent gradient explosion
    """
    self.lr = lr
    self.batch_size = batch_size
    self.nb_epochs = nb_epochs
    self.clip_threshold = clip_threshold
    self.prev_gradient_norm = None # previously computed gradient norm for clipping logging
    self.momentum = momentum
    self.velW, self.velb = None, None # velocities for momentum

  def reset_velocities(self):
    self.velW, self.velb = None, None

  def _compute_norm(self, dW, db):
    total = 0.0
    for gradient in dW + db: # list concat
      total += np.sum(gradient ** 2) # sum of squares
    return np.sqrt(total) # norm

  def _clip(self, dW, db):
    norm = self._compute_norm(dW, db)
    self.prev_gradient_norm = norm

    if norm > self.clip_threshold:
      dW = [gradient * (self.clip_threshold / norm) for gradient in dW]
      db = [gradient * (self.clip_threshold / norm) for gradient in db]

    return dW, db

  def run(self, model: MLP, dW, db):
    """
    Apply a single updateto model.weights and model.biases
    Includes rolling momentum for faster convergence!
    """
    if self.clip_threshold is not None:
      dW, db = self._clip(dW, db)

    if self.velW is None or self.velb is None:
      self.velW = [np.zeros_like(w) for w in model.weights]
      self.velb = [np.zeros_like(b) for b in model.biases]

    for i in range(model.nb_layers):
      # momentum gathering
      self.velW[i] = self.momentum * self.velW[i] + dW[i]
      self.velb[i] = self.momentum * self.velb[i] + db[i]

      # weight updates, by momentum instead of raw gradients
      model.weights[i] -= self.lr * self.velW[i]
      model.biases[i] -= self.lr * self.velb[i]
