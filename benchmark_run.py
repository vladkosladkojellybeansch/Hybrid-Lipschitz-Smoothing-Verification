import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter, FuncAnimation
from robust_nn import LipschitzNeuralNetwork
from advanced_smoothing import AdvancedRandomizedSmoother

np.random.seed(42)
X = np.random.uniform(-1, 1, (200, 10))
y = (np.sum(X**2, axis=1) > 0.8).astype(int)

base_nn = LipschitzNeuralNetwork(input_dim=10, hidden_dim=20, clip_value=0.05)
base_nn.fit(X, y, epochs=50, lr=0.01)

fig, ax = plt.subplots(figsize=(10, 6))
sigmas = np.linspace(0.05, 0.3, 10)
sigmas_np = np.array(sigmas)

ax.set_xlim(0.04, 0.31)
ax.set_ylim(-0.02, 0.5)
ax.set_title('Dynamic Certified Robustness Bounds (Sinusoidal Data Shift)', fontsize=12, fontweight='bold')
ax.set_xlabel('Smoothing Parameter $\sigma$ / Adversarial Noise Scale')
ax.set_ylabel('Certified Protection Radius ($L_2$ Space)')
ax.grid(True, linestyle=':', alpha=0.6)

line_smooth, = ax.plot([], [], color='darkorange', linewidth=2.5, marker='s', label="Cohen's Randomized Smoothing $L_2$ Radius")
line_lipschitz, = ax.plot([], [], color='teal', linestyle='--', linewidth=2, label="Deterministic 1-Lipschitz Structural Bound")
ax.legend(loc='lower right')

frames_data = []
print("Изчисляване на кадрите по синусоидален закон...")

for idx, frame in enumerate(np.linspace(0, 2 * np.pi, 30)):
    sin_shift = np.sin(frame) * 0.8  
    x_dynamic = X[15:16].copy()      
    x_dynamic[0, :5] += sin_shift    
    
    smoothing_radii = []
    _, _, base_margin = base_nn.certify_robustness(x_dynamic, adversarial_perturbation_norm=0.0)
    
    for s in sigmas:
        test_smoother = AdvancedRandomizedSmoother(base_classifier=base_nn, sigma=s)
        _, r_smooth = test_smoother.certify_robustness(x_dynamic, n_samples=150)
        smoothing_radii.append(r_smooth)
        
    frames_data.append((list(smoothing_radii), base_margin))

print("Сглобяване на анимацията...")
def animate(i):
    smoothing_radii, base_margin = frames_data[i]
    line_smooth.set_data(sigmas, smoothing_radii)
    line_lipschitz.set_data(sigmas, [base_margin] * len(sigmas))
    return line_smooth, line_lipschitz

ani = FuncAnimation(fig, animate, frames=len(frames_data), blit=True)
writer = PillowWriter(fps=12)
ani.save('robustness_animation.gif', writer=writer)
plt.close()
print("Успех! Файлът 'robustness_animation.gif' е готов.")
