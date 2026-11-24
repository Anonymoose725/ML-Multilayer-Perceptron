"""scikit-learn wrapper around the MLP for GridSearchCV."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from src.optimizer import GradientDescent
from src.mlp import MLP
from src.config import SEED


class Wrapper(BaseEstimator, ClassifierMixin):
  def __init__(self, hidden=(256, 256), a_func='relu', reg=None, lam=0.0, dropout=0.0, lr=0.1, batch_size=128, nb_epochs=10):
    # scikitlearn rules says the values can't change in the initializer and they have to have the same names as arguments
    self.hidden, self.a_func, self.reg, self.lam, self.dropout = hidden, a_func, reg, lam, dropout # regularization + config
    self.lr, self.batch_size, self.nb_epochs = lr, batch_size, nb_epochs                           # fitting + descent

  def fit(self, X, y):
    # node self.model_ is a private convention, like helpers are def _func()
    self.model_ = MLP(hidden=self.hidden, a_func=self.a_func, reg=self.reg, lam=self.lam, dropout=self.dropout, seed=SEED)
    optimizer = GradientDescent(lr=self.lr, batch_size=self.batch_size, nb_epochs=self.nb_epochs)
    self.model_.fit(X, y, 10, optimizer, logging=False)
    self.classes_ = np.unique(y) # returns sorted unique classes in an array

    return self # skl expectation is that fit returns self

  def predict(self, X):
    return self.model_.predict(X)
