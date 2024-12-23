import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal

# Define the simple quadratic function and its gradient
def func(x, y):
    return x**2 + y**2

def grad(x, y):
    return 2 * x, 2 * y

num_samples = 5000

# Initialize starting point
np.random.seed(42)

# Generate the grid for contour plot
x_vals = np.linspace(-5, 5, 1000)
y_vals = np.linspace(-5, 5, 1000)
X, Y = np.meshgrid(x_vals, y_vals)
Z = func(X, Y)

# Parameters for MALA
initial_point = [-4.4, 4.4]
x, y = initial_point
samples = []
acceptance_probs = []

x_mean = np.mean(x_vals)
y_mean = np.mean(y_vals)
x_std = np.std(x_vals)
y_std = np.std(y_vals)

# Step size and decay parameters
epsilon_0 = 0.95  # Initial step size
decay_rate = 0.0001

# MALA Sampling
for t in range(num_samples):
    # Apply step size decay
    epsilon = epsilon_0 / (1 + decay_rate * t)
    sigma = np.sqrt(epsilon) * 2

    # Calculate gradient at current point
    grad_x, grad_y = grad(x, y)

    # Propose a new point
    x_proposed = x - (epsilon / 2) * grad_x + np.random.normal(0, sigma)
    y_proposed = y - (epsilon / 2) * grad_y + np.random.normal(0, sigma)

    # Compute acceptance probability
    current_U = func(x, y)
    proposed_U = func(x_proposed, y_proposed)

    # Compute transition probabilities
    q_forward = multivariate_normal.pdf(
        [x_proposed, y_proposed], mean=[x - (epsilon / 2) * grad_x, y - (epsilon / 2) * grad_y], cov=epsilon * np.eye(2)
    )
    q_backward = multivariate_normal.pdf(
        [x, y], mean=[x_proposed - (epsilon / 2) * grad(x_proposed, y_proposed)[0],
                      y_proposed - (epsilon / 2) * grad(x_proposed, y_proposed)[1]], cov=epsilon * np.eye(2)
    )

    alpha = min(1, np.exp(current_U - proposed_U) * (q_backward / q_forward))
    acceptance_probs.append(alpha)  # Track acceptance probability

    # Accept or reject
    if np.random.uniform(0, 1) < alpha:
        x, y = x_proposed, y_proposed  # Accept the new state

    samples.append((x, y))

samples = np.array(samples)

# Contour plot of the quadratic function
plt.figure(figsize=(6.4, 4.8))
contour = plt.contourf(X, Y, Z, levels=50, cmap='inferno')
plt.title('Contour Plot of Quadratic Function with MALA Samples')
plt.colorbar(contour)
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.3, s=10, color="white", label="Samples")
plt.xlabel("x")
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.ylabel("y")
plt.legend()
plt.savefig("Samples_QuadraticFunction_MALA_LargeStep.png")
plt.show()

x_dist = np.random.normal(x_mean, x_std, 1000)
y_dist = np.random.normal(y_mean, y_std, 1000)

# Overlayed KDE for Slope Samples and X values
plt.figure(figsize=(6.4, 4.8))
sns.kdeplot(samples[:, 0], fill=True, color="darkblue", alpha=0.8, label="Slope Samples")
sns.kdeplot(x_dist, fill=True, color="lightblue", alpha=0.8, label="Slope True Values")
plt.title("Slope Samples and Slope True Values")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig("KDE_Slope_XOverlay_MALA_LargeStep.png")
plt.show()

# Overlayed KDE for Intercept Samples and Y values
plt.figure(figsize=(6.4, 4.8))
sns.kdeplot(samples[:, 1], fill=True, color="darkgreen", alpha=0.8, label="Intercept Samples")
sns.kdeplot(y_dist, fill=True, color="lightgreen", alpha=0.8, label="Intercept True Values")
plt.title("Intercept Samples and Intercept True Values")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig("KDE_Intercept_YOverlay_MALA_LargeStep.png")
plt.show()

# Print the average acceptance probability
avg_acceptance = np.mean(acceptance_probs)
print(f"Average Acceptance Probability: {avg_acceptance:.4f}")
