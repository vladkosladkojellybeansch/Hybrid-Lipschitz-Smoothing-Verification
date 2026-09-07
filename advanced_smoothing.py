import numpy as np
from scipy.stats import norm

class AdvancedRandomizedSmoother:
    """
    Certified Adversarial Robustness via Randomized Smoothing (Cohen et al., 2019 / Hybrid 2025).
    """
    def __init__(self, base_classifier, sigma=0.12):
        self.base_model = base_classifier
        self.sigma = sigma

    def _sample_under_noise(self, x, num_samples=1000):
        noise = np.random.normal(0, self.sigma, (num_samples, x.shape[1]))
        perturbed_inputs = x + noise
        predictions = []
        for p_in in perturbed_inputs:
            pred = self.base_model.forward(p_in.reshape(1, -1))
            predictions.append(1 if pred > 0.5 else 0)
        return np.array(predictions)

    def certify_robustness(self, x, n_samples=1000, alpha=0.001):
        predictions = self._sample_under_noise(x, num_samples=n_samples)
        counts = np.bincount(predictions, minlength=2)
        top_class = np.argmax(counts)
        n_top = counts[top_class]
        p_a = np.clip(n_top / n_samples, 0.0, 0.999) # Числено подрязване за елиминиране на 'inf'
        
        if p_a > 0.5:
            radius = self.sigma * norm.ppf(p_a)
        else:
            radius = 0.0  
        return top_class, float(radius)
