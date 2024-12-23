# Reimport necessary libraries
import numpy as np
import matplotlib.pyplot as plt

# Redefine the quadratic function and its gradient
def func(x, y):
    return x**2 + y**2

def grad(x, y):
    return 2 * x, 2 * y

# Generate the grid for contour plot
x_vals = np.linspace(-5, 5, 100)
y_vals = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = func(X, Y)

# Function to perform SGLD sampling and plot results
def sgld_sampling_and_plot(samples_count, step_size, title_suffix):
    initial_point = [-4.4, 4.4]
    x, y = initial_point  # Reset starting point for each run
    samples = []

    for t in range(samples_count):
        # Keep the initial step size constant for first three sweeps
        if t < 3 * len(X.flatten()):  # Sweeping the dataset three times
            current_step_size = 0.1
        else:
            current_step_size = step_size

        noise_scale = np.sqrt(current_step_size)
        grad_x, grad_y = grad(x, y)
        x += -current_step_size * grad_x + np.random.normal(0, noise_scale)
        y += -current_step_size * grad_y + np.random.normal(0, noise_scale)
        samples.append((x, y))

    samples = np.array(samples)

    # Contour plot
    plt.figure(figsize=(6.4, 4.8))
    contour = plt.contourf(X, Y, Z, levels=50, cmap='inferno')
    plt.title(f'Contour Plot of Quadratic Function with {samples_count} Samples {title_suffix}')
    plt.colorbar(contour)
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.3, s=10, color="white", label="Samples")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

# Sweep number of samples
for num_samples in [50, 500, 5000]:
    sgld_sampling_and_plot(num_samples, 0.1, "(Fixed Step Size)")

# Sweep learning rate with fixed number of samples
for step_size in [0.000000001, 0.1, 10000.0]:
    sgld_sampling_and_plot(500, step_size, f"(Step Size = {step_size})")
