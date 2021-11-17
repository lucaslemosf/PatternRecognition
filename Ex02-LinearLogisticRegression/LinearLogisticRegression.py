# Pattern Recognition
# Programming Exercise 2 - 2021-11-17
# Linear Logistic Regression

# Using sklearn.linear_model.SGDClassifier instead of sklearn.linear_model.Logistic Regression
# since with it we can define Learning rate

from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


class LinearLogisticRegression:

    def __init__(self, learningRate=0.5, maxIterations=100):
        self.learningRate = learningRate
        self.maxIterations = maxIterations
        X, y = load_iris(return_X_y=True)
        self.clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=self.maxIterations, learning_rate=self.learningRate,
                                                            tol=1e-3))
        self.fit(X, y)
        self.predict(X, y)


    def fit(self, X, y):
        self.clf.fit(X, y)


    def gFunc(self, X, theta):
        pass


    def predict(self, X):
        print(self.clf.predict(X))
        pass
