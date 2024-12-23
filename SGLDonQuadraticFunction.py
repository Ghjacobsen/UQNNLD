import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define the simple quadratic function and its gradient
def func(x, y):
    return x**2 + y**2

def grad(x, y):
    return 2 * x, 2 * y

num_samples = 1000

# Initialize starting point
np.random.seed(42)

# Generate the grid for contour plot
x_vals = np.linspace(-5, 5, 100)
y_vals = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = func(X, Y)

x_mean = np.mean(x_vals)
y_mean = np.mean(y_vals)
x_std = np.std(x_vals)
y_std = np.std(y_vals)
print(x_mean, y_mean, x_std, y_std)

# Updated SGLD parameters with fixed starting point and learning rate decay
initial_point = [-20,-20]
x, y = initial_point
samples = []

# Learning rate decay parameters
initial_step_size = 0.1
decay_rate = 0.001

# Perform SGLD sampling with learning rate decay
for t in range(num_samples):
    step_size = initial_step_size / (1 + decay_rate * t)  # Decayed step size
    noise_scale = np.sqrt(step_size) * 2
    grad_x, grad_y = grad(x, y)
    x += -step_size * grad_x #+ np.random.normal(0, noise_scale)
    y += -step_size * grad_y #+ np.random.normal(0, noise_scale)
    samples.append((x, y))

samples = np.array(samples)

# Adjusted limits for contour plot to "zoom out"
zoom_factor = 2
x_vals_zoomed = np.linspace(-5 * zoom_factor, 5 * zoom_factor, 100)
y_vals_zoomed = np.linspace(-5 * zoom_factor, 5 * zoom_factor, 100)
X_zoomed, Y_zoomed = np.meshgrid(x_vals_zoomed, y_vals_zoomed)
Z_zoomed = func(X_zoomed, Y_zoomed)

# Contour plot of the quadratic function
plt.figure(figsize=(6.4, 4.8))
contour = plt.contourf(X_zoomed, Y_zoomed, Z_zoomed, levels=50, cmap='inferno')
#plt.title('Contour Plot of Quadratic Function with SGD Samples')
plt.colorbar(contour)
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.3, s=10, color="white", label="Samples")
plt.xlabel("x")
plt.xlim(-5 * zoom_factor, 5 * zoom_factor)
plt.ylim(-5 * zoom_factor, 5 * zoom_factor)
plt.ylabel("y")
plt.legend()
plt.savefig("Samples_QuadraticFunction_Zoomed_SGD.png")
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
plt.savefig("KDE_Slope_XOverlay.png")
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
plt.savefig("KDE_Intercept_YOverlay.png")
plt.show()