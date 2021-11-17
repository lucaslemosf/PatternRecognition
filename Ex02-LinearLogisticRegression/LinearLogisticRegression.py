# Pattern Recognition
# Programming Exercise 2 - 2021-11-17
# Linear Logistic Regression

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression


def __init__(self, learningRate=0.5, maxIterations=100):
    self.learningRate = learningRate
    self.maxIterations = maxIterations
    X, y = load_iris(return_X_y=True)
    clf = LogisticRegression(random_state=0, max_iter=self.maxIterations, )

def fit(self, X, y):
    pass


def gFunc(self, X, theta):
    pass


def predict(self, X):
    pass
