import numpy as np

class LipschitzNeuralNetwork:
    """
    1-Lipschitz Neural Network with hard weight clipping constraints (L <= 1).
    """
    def __init__(self, input_dim, hidden_dim, clip_value=0.05):
        self.clip_value = clip_value
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, 1) * 0.01
        self.b2 = np.zeros((1, 1))
        self._clip_weights()

    def _clip_weights(self):
        self.W1 = np.clip(self.W1, -self.clip_value, self.clip_value)
        self.W2 = np.clip(self.W2, -self.clip_value, self.clip_value)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.maximum(0, self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return self.z2

    def fit(self, X, y, epochs=500, lr=0.05):
        m = X.shape[0]
        y = y.reshape(-1, 1)
        for epoch in range(epochs):
            scores = self.forward(X)
            dloss_dscores = 2 * (scores - y) / m
            dW2 = np.dot(self.a1.T, dloss_dscores)
            db2 = np.sum(dloss_dscores, axis=0, keepdims=True)
            da1 = np.dot(dloss_dscores, self.W2.T)
            dz1 = da1 * (self.z1 > 0)
            dW1 = np.dot(X.T, dz1)
            db1 = np.sum(dz1, axis=0, keepdims=True)
            self.W1 -= lr * dW1
            self.b1 -= lr * db1
            self.W2 -= lr * dW2
            self.b2 -= lr * db2
            self._clip_weights()

    def certify_robustness(self, x, adversarial_perturbation_norm):
        logit = self.forward(x)
        prediction = 1 if logit > 0.5 else 0
        margin = np.abs(logit - 0.5)
        certified_margin = margin - adversarial_perturbation_norm
        return int(prediction), bool(certified_margin > 0), float(np.squeeze(certified_margin))
