import numpy as np
from sklearn.ensemble import RandomForestClassifier

class IPLPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def train(self):
        X = np.random.rand(500, 7)
        y = np.random.randint(0, 2, 500)
        self.model.fit(X, y)

    def predict(self, X):
        pred = self.model.predict(X)[0]
        prob = self.model.predict_proba(X)[0]
        return pred, prob
