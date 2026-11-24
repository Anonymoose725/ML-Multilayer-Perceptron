"""From-scratch NumPy MLP: layers, activations, loss, backprop."""
import numpy as np
import time
from src.config import SEED
from src.preprocessing import to_arrays


logistic = lambda z: 1./ (1 + np.exp(-z))


def softmax(z): # 2-dimensional for batches
  z = z - z.max(axis=1, keepdims=True) # numerically stable version subtracting row before exponentiating
  e = np.exp(z)
  return e / e.sum(axis=1, keepdims=True)


def generate_a_func(name, l=0.01):
  """
  Gives a callable elementwise function to apply on numpy arrays
  Inputs:
  name:   'logistic' | 'relu' | 'leaky_relu' | 'tanh'
  l:      hyperparameter to tune for leaky relu, shrinking factor for z < 0
  Output:
  (f, f_prime):   elementwise function and its derivative to apply on np.array
  """
  if name == 'logistic':
      return (lambda z: logistic(z),                    # 1 / 1+e^-z
              lambda z: np.multiply(logistic(z), 1 - logistic(z))) # df/dx = f(x)(1-f(x))
  elif name == 'relu':
      return (lambda z: np.maximum(0, z),            # send to 0
              lambda z: (z > 0).astype(z.dtype))     # { 1 if z > 0; 0 otherwise }
  elif name == 'leaky_relu':
      return (lambda z: np.where(z > 0, z, l * z), # scale by a (hyperparameter to tune!)
              lambda z: np.where(z > 0, 1.0, l))   # { 1 if z > 0; a otherwise }
  elif name == 'tanh':
      return (np.tanh,                               # s shaped
              lambda z: 1.0 - np.tanh(z) ** 2)       # sech^2 = 1-tanh
  else:
    raise ValueError("Pick one of 'logistic' / 'relu' / 'leaky_relu' / 'tanh'")


def evaluate_acc(y, y_hat):
  """
  Evaluates model accuracy to assignment spec
  """
  return np.mean(y == y_hat)


def cross_entropy(y, p, eps=1e-12):
  """
  Computes cross-entropy loss

  Inputs:
  y:    integer labels, shape (N,)
  p:    softmax probabilities, shape (N, C)
  eps:  small noise
  """
  N = y.shape[0]
  p_true = p[np.arange(N), y]
  # picks the true class out of each. avoids one-hot encoding
  # arange evenly spaces values so we can pick them out as classes
  return -np.mean(np.log(np.clip(p_true, eps, 1.0))) # keeps within bounds


class MLP:

  def __init__(self, hidden=(256, 256), a_func='relu', leaky_factor=0.0, reg=None, lam=0.0, dropout=0.0, seed=SEED):
    """
    Initialize a K-layer neural network multilayer perception

    x -> [W1,b1] -> σ -> ... -> [W_K,b_K] -> σ -> [W_out,b_out] -> softmax

    Inputs:
    hidden:       hidden units per layer, ex. (256, 256) meaning 2 hidden layers of 256 units each.
                  an empty set () indicates 0 hidden layers, reducing MLP to softmax regression
    activation:   activation function of each layer and hidden unit respectively
                  'relu' | 'logistic' | 'leaky_relu' | 'tanh'
    leaky_factor: factor to scale leaky relu by, when selected. 0 by default (reduces to relu)
    reg:          regularization to apply during weight updates
                  'L1' | 'L2' | None
    lam:          regularization strength, lambda. Higher indicates greater regularization strength
    dropout:      0 <= dropout < 1, rate of hidden unit dropout to aid in generalization. 0 reduces to no dropout.

    Call _init_weights(D, C) to initialize the weights and biases before fitting
    """
    self.hidden = list(hidden)
    self.nb_layers = len(hidden) + 1 # +1 for output layer
    self.activation = a_func
    self.a_func, self.da_func = generate_a_func(a_func, l=leaky_factor) # callable elementwise functions
    self.reg = reg
    self.lam = lam
    self.rng = np.random.default_rng(seed) # call again
    self.dropout = np.clip(dropout, 0.0, 0.999)
    self.drop_rng = np.random.default_rng(seed+1) # different from rng
    self.weights = []
    self.biases = []

    # weight initialization: Kaiming/He for ReLU activation, Xavier-Glorot for logistic or tanh
    # helps eliminate gradient vanishing since loss is non-convex, and by avoiding the symmetry problem
    # we'll leave them empty for now since the shape of the dataset isnt known until runtime (since we're augmenting, it changes shape)
    # see init_weights(D, C) below and call it!

  def init_weights(self, D, C):
    """
    Initialize weights of an MLP to be non-zero to avoid symmetry and gradient vanishing.
    Optimal initialization is dependent on the activation function of hidden layers

    Inputs:
    D:    Number of features in the input
    C:    Number of classes in the output

    """
    self.weights, self.biases = [], [] # start from nothing in case we're reinitializing
    layer_sizes = [D] + self.hidden + [C] # layer sizes from input to output, like [784, 256, 256, 10] in default case

    # weights take inputs to next layer
    layer_dims = list(zip(layer_sizes[:-1], layer_sizes[1:])) # [(784, 256), (256, 256), (256, 10)] in default case
    nb_layers = len(layer_dims)

    for layer in range(nb_layers):
      D_in, D_out = layer_dims[layer]
      is_last = layer == nb_layers - 1 # last layer is output

      # check which activation layer used
      if self.activation in ('relu', 'leaky_relu') or is_last: # putting last layer through Kaiming compensates for relu sending many logits to zero
        variance = 2 / D_in # Kaiming/He
      elif self.activation in ('logistic', 'tanh'):
        variance = 2 / (D_in + D_out) # Xavier-Glorot
      else:
        raise ValueError("Something went wrong in initialization!") # technically avoidable because it's needed for init

      # use dynamic variances for random initialization of weights
      std = np.sqrt(variance)
      W = self.rng.normal(loc = 0.0, scale = std, size = (D_in, D_out))
      b = np.zeros(D_out) # ok to leave biases at zero

      self.weights.append(W)
      self.biases.append(b)


  def forward(self, x, training=False):
    """Return softmax logits (N, C) and list (a_prev, z) for backprop"""
    # loop over layers. for each hidden layer, store layers input 'a_prev' and preactivation 'z'
    # then a = self.a_func(z)
    # last layer has no activation. output logits as softmax
    # keep running precomputed values for backprop: list(a_prev, z) per layer
    a = x
    a_prevs = []
    for i in range(self.nb_layers):
      a_prev = a
      z = a_prev @ self.weights[i] + self.biases[i] # @ is dot product
      # a_prevs.append((a_prev, z))

      dropout_mask = None


      if i == self.nb_layers - 1: # last layer is output and is just identity
        a = z
      else:
        a = self.a_func(z) # activate z
        if self.dropout > 0:
          if training: # dropout, only during training, not during predictions
            dropout_mask = (self.drop_rng.random(a.shape) >= self.dropout).astype(a.dtype) # if it exceeds dropout rate, drop
            a = a * dropout_mask # mask array with dropouts per layer
          else:
            a = a * (1-self.dropout) # scale by the non-drop probability to make up for lost weight

      a_prevs.append((a_prev, z, dropout_mask)) # dropout_mask = None when dropout=0.0

    return a, a_prevs # last ones are logits


  def backprop(self, p, y, a_prevs):
    """
    Applies the chain rule from loss back to every W and b, reusing previous computations at each layer

    Inputs:
    p:        softmax probabilities (N,C)
    y:        integer labels (N,)
    a_prevs:  (a_prev, z) for each layer

    Outputs:
    (dW, db): gradients, one array per layer in forward layer (since for loop works backwards)
    """
    # gradients
    dW = [None] * self.nb_layers
    db = [None] * self.nb_layers

    # we do output dZ first
    N = y.shape[0]
    dZ = p.copy()
    dZ[np.arange(N), y] -= 1 # removes the one-hot matrix to obtain p-Y where Y is one-hot, i.e p-1 when true
    dZ = dZ / N # normalize as mean average

    for i in range(self.nb_layers -1, -1, -1): # backwards
      (a_prev, z, dropout_mask) = a_prevs[i] # dropout_mask = None if dropout=0.0

      # compute gradients
      dW[i] = a_prev.T @ dZ
      db[i] = dZ.sum(axis=0) # just the firsts

      # regularization penalty on weights
      if self.reg == 'L2':    # sends closer to 0
        dW[i] += (self.lam * self.weights[i])
      elif self.reg == 'L1':  # sends exactly to 0
        dW[i] += (self.lam * np.sign(self.weights[i]))

      if i > 0: # for all but output
        z_prev = a_prevs[i-1][1] # (a_prev, ->z)
        dropout_prev = a_prevs[i-1][2]
        dZ = dZ @ self.weights[i].T
        if dropout_prev is not None:
          dZ = dZ * dropout_prev # drop units

        dZ = dZ * self.da_func(z_prev) # apply derivative activation function chain rule

    return dW, db

  def fit(self, X, y, C, optimizer, X_val=None, y_val=None, initialize_weights=True, logging=True, train_dataset=None): # optimizer contains hyperparameters
    """
    Fit an MLP with backprop and minibatch GD

    Inputs:
    X:                    input features, shape (N, D)
    y:                    labels, shape (N,)
    C:                    number of classes
    optimizer:            holds nb_epochs, batch_size, learn_rate, and the update rule optimizer.run
    X_val, y_val:         optional validation data, evaluated at the end of each epoch
    initialize_weights:   if True, start with fresh weights, otherwise continue training
    logging:              prints a progress line per epoch
    train_dataset:        dataset used to generate a newly augmented set at start of each epoch. none by default indicates no augmenting

    Output:
    stats:  dict of per-epoch lists (train_loss, train_acc, val_loss, val_acc, elapsed)
            where 'time' is cumulative training time so-far
    """
    N, D = X.shape
    if initialize_weights:
      self.init_weights(D, C) # start with fresh weight

    rng = np.random.default_rng(SEED) # for shuffling

    stats = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'elapsed': []}
    time_taken = 0.0

    def _produce_batches(N, batch_size, rng):
      shuffle = rng.permutation(N) # shuffle
      for i in range(0, N, batch_size): # step by batch_size until N
        yield shuffle[i:i+batch_size]

    for epoch in range(optimizer.nb_epochs):
      time_start = time.perf_counter()

      # check if augmenting
      if train_dataset is not None:
        X_epoch, y_epoch = to_arrays(train_dataset)
      else:
        X_epoch, y_epoch = X, y

      for batch in _produce_batches(len(X_epoch), optimizer.batch_size, rng):
        Xs, ys = X_epoch[batch], y_epoch[batch] # matrices of shape (batch_size, D), (batch_size,)

        # forward pass + softmax
        logits, a_prevs = self.forward(Xs, training=True)
        p = softmax(logits) # probabilities output

        # compute gradients with backpropagation, note regularization is done here
        dW, db = self.backprop(p, ys, a_prevs)

        # step forward in minibatch SGD
        optimizer.run(self, dW, db)

      time_taken += time.perf_counter() - time_start # time for each epoch

      # collect metrics
      validating = X_val is not None and y_val is not None
      train_loss, train_acc = self.eval(X, y)
      stats['train_loss'].append(train_loss)
      stats['train_acc'].append(train_acc)
      stats['elapsed'].append(time_taken)
      if validating:
        validate_loss, validate_acc = self.eval(X_val, y_val)
        stats['val_loss'].append(validate_loss)
        stats['val_acc'].append(validate_acc)

      if logging and (epoch + 1) % 10 == 0: # print every 10th
        print(f"Epoch #{epoch+1}/{optimizer.nb_epochs}:  "
              f"loss {train_loss:.4f} | acc {train_acc:.4f}")
        if validating:
          print(f"loss {validate_loss:.4f} | acc {validate_acc:.4f}")

    return stats


  def predict(self, X):
    """Returns a classification prediction of input X by the MLP neural network"""
    logits, _ = self.forward(X)
    return logits.argmax(axis=1) # argmax of softmax = highest probability class

  def eval(self, X, y):
    """
    Mean CE and accuracy on a full dataset

    Inputs:
    X:    inputs, shape (N, D)
    y:    labels, shape (N,)

    Output:
    (mean CE, accuracy): statistical measures pasing over the entire dataset (no regularization penalty)
    """
    logits,_ = self.forward(X)
    p = softmax(logits)

    CE = cross_entropy(y, p)
    acc = np.mean(logits.argmax(axis=1) == y) # measure of correct predictions

    return CE, acc
