import numpy as np
import matplotlib.pyplot as plt

# Define the simple quadratic function and its gradient
def func(x, y):
    return x**2 + y**2

def grad(x, y):
    return 2 * x, 2 * y

# Generate the grid for contour and 3D plots
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = func(X, Y)

# Plot the Contour Plot
plt.figure(figsize=(6.4, 4.8))
contour = plt.contourf(X, Y, Z, levels=50, cmap='inferno')
plt.title('Contour Plot of Quadratic Function')
plt.colorbar(contour)
plt.savefig("ContourPlotOfQuadratic.png")
plt.close()

fig = plt.figure(figsize=(6.4, 4.8))
ax3d = fig.add_subplot(111, projection='3d')

# Plot the surface
ax3d.plot_surface(X, Y, Z, cmap='inferno', alpha=0.8)

# Simplify axes
ax3d.set_xticks([-5, 5])  # Min and max only
ax3d.set_yticks([-5, 5])
ax3d.set_zticks([0, 50])

# Add axis labels
ax3d.set_xlabel("X-axis")
ax3d.set_ylabel("Y-axis")
ax3d.set_zlabel("Z-axis")

# Lighten the grid lines
ax3d.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

# Rotate to a dynamic perspective
ax3d.view_init(elev=30, azim=120)

# Add a title
ax3d.set_title('3D Visualization of Quadratic Function')

# Save and display the improved plot
plt.tight_layout()
plt.savefig("3DPlotOfQuadratic.png")
#plt.show()
plt.close()

# Perform gradient descent
start_point = np.array([4.0, 4.0])  # Starting point
learning_rate = 0.1
iterations = 30
points = [start_point]

current_point = start_point
for _ in range(iterations):
    gradient = np.array(grad(*current_point))
    next_point = current_point - learning_rate * gradient
    points.append(next_point)
    current_point = next_point
final_point = current_point
print(final_point)
# Convert gradient descent path to arrays
path_x = [p[0] for p in points]
path_y = [p[1] for p in points]
path_z = [func(p[0], p[1]) for p in points]

# Plot a single contour plot with the gradient descent path
plt.figure(figsize=(6.4, 4.8))

""" contour = plt.contourf(X, Y, Z, levels=50, cmap='inferno')
plt.plot(path_x, path_y, color='cyan', marker='o', markersize=5, label='Gradient Descent Path')
plt.title('Contour Plot with Gradient Descent Path')
plt.legend()
plt.colorbar(contour)

# Save the plot
plt.savefig("CountourPlotWithGradientDescent.png")
plt.show() """

def noisy_grad(x, y, noise_level=1.0):
    gx, gy = grad(x, y)
    noise_x = np.random.normal(0, noise_level)
    noise_y = np.random.normal(0, noise_level)
    return gx + noise_x, gy + noise_y

# Perform SGD with noise
sgd_points_high_noise = [start_point]
current_point = start_point
decay_rate = 0.001
for _ in range(iterations):
    gradient = np.array(noisy_grad(*current_point, noise_level=3.0))  # Noise level
    next_point = current_point - learning_rate * gradient
    learning_rate = learning_rate / (1 + decay_rate * iterations)
    sgd_points_high_noise.append(next_point)
    current_point = next_point

# Convert high-noise SGD path to arrays
sgd_high_noise_path_x = [p[0] for p in sgd_points_high_noise]
sgd_high_noise_path_y = [p[1] for p in sgd_points_high_noise]

# Create a figure with both gradient descent and SGD
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Contour plot with gradient descent path
contour1 = axs[0].contourf(X, Y, Z, levels=50, cmap='inferno')
axs[0].plot(path_x, path_y, color='cyan', marker='o', markersize=5, label='Gradient Descent Path')
axs[0].set_title('Contour Plot with Gradient Descent Path')
axs[0].legend()
plt.legend(loc='upper left', bbox_to_anchor=(0, 1))
fig.colorbar(contour1, ax=axs[0])

# Contour plot with high-noise SGD path
contour2 = axs[1].contourf(X, Y, Z, levels=50, cmap='inferno')
axs[1].plot(sgd_high_noise_path_x, sgd_high_noise_path_y, color='green', marker='o', markersize=5, label='SGD Path')
axs[1].set_title('Contour Plot with SGD Path')
axs[1].legend()
fig.colorbar(contour2, ax=axs[1])

# Save the combined plot
plt.tight_layout()
plt.legend(loc='upper left', bbox_to_anchor=(0, 1))
plt.savefig("CountourGrad&SGD.png")
#plt.show()
plt.close()

# Define the new function in 3D based on sin(x) * 2x with the described minima
def new_function_3d(x, y):
    return (np.sin(x) * 2 * x) + (np.sin(y) * 2 * y)

def new_function_3d_grad(x, y):
    grad_x = 2 * (np.sin(x) + x * np.cos(x))
    grad_y = 2 * (np.sin(y) + y * np.cos(y))
    return grad_x, grad_y

def noisy_grad(x, y, noise_level=1.0):
    grad_x, grad_y = new_function_3d_grad(x, y)
    noise_x = np.random.normal(0, noise_level)
    noise_y = np.random.normal(0, noise_level)
    return grad_x + noise_x, grad_y + noise_y

# Parameters
learning_rate = 0.05
iterations = 50

# Generate the grid for the new function in 3D
x = np.linspace(-2, 6, 200)
y = np.linspace(-2, 6, 200)
X, Y = np.meshgrid(x, y)
Z = new_function_3d(X, Y)

# Starting point
start_point = np.array([2.0, 2.0])

# Perform standard Gradient Descent on the simplified function
gd_points = [start_point]
current_point = start_point
decay_rate = 0.001
for _ in range(iterations):
    gradient = np.array(new_function_3d_grad(*current_point))
    next_point = current_point - learning_rate * gradient
    gd_points.append(next_point)
    current_point = next_point

final_gd = current_point
final_gd_z = new_function_3d(final_gd[0], final_gd[1])
print(final_gd_z)

gd_path_x = [p[0] for p in gd_points]
gd_path_y = [p[1] for p in gd_points]
gd_path_z = [new_function_3d(p[0], p[1]) for p in gd_points]

# Perform SGD with noise on the simplified function
sgd_points_high_noise = [start_point]
current_point = start_point
for _ in range(iterations):
    gradient = np.array(noisy_grad(*current_point, noise_level=3.0))  # Increased noise level
    next_point = current_point - learning_rate * gradient
    learning_rate = learning_rate / (1 + decay_rate * iterations)
    sgd_points_high_noise.append(next_point)
    current_point = next_point

final_sgd = current_point
final_sgd_z = new_function_3d(final_sgd[0], final_sgd[1])
print(final_sgd_z)

sgd_high_noise_path_x = [p[0] for p in sgd_points_high_noise]
sgd_high_noise_path_y = [p[1] for p in sgd_points_high_noise]
sgd_high_noise_path_z = [new_function_3d(p[0], p[1]) for p in sgd_points_high_noise]


plt.figure(figsize=(6.4, 4.8))

# Contour plot with both paths
contour = plt.contourf(Y, X, Z, levels=50, cmap='inferno')
plt.plot(gd_path_x, gd_path_y, color='cyan', marker='o', markersize=5, label='Gradient Descent Path')
plt.plot(sgd_high_noise_path_x, sgd_high_noise_path_y, color='green', marker='o', markersize=5, label='SGD Path')
plt.title('Contour Plot with Gradient Descent and SGD Paths')
plt.legend()
plt.legend(loc='upper left', bbox_to_anchor=(0, 1))
plt.savefig("SGDvsGD.png")
plt.colorbar(contour)

plt.tight_layout()
plt.show()

# Plotting the posterior density
# Plotting the posterior density
plt.figure(figsize=(8, 6))
contour = plt.contourf(X, Y, Z, levels=100, cmap='inferno', alpha=0.8)
colorbar = plt.colorbar(contour, label='Posterior Density')

# Plot the paths for GD and SGD
plt.plot(gd_path_x, gd_path_y, color='cyan', marker='o', markersize=5, label='Gradient Descent Path')
plt.plot(sgd_high_noise_path_x, sgd_high_noise_path_y, color='green', marker='o', markersize=5, label='SGD Path')

# Mark final points with black cross
plt.scatter(final_sgd[0], final_sgd[1], color='black', marker='x', s=100, zorder=5)
plt.scatter(final_gd[0], final_gd[1], color='black', marker='x', s=100, zorder=5, label='Final Points')

plt.scatter(2, 2, color='Red', marker='x', s=100, zorder=5, label='Start Point')

# Add text boxes for final points near the color bar
# Add final point text boxes at the bottom right of the plot
""" plt.text(4.5, -1.5, f"SGD Final:\n({final_sgd[0]:.2f}, {final_sgd[1]:.2f}, Z={int(final_sgd_z)})",
         bbox=dict(facecolor='white', edgecolor='green', boxstyle='round,pad=0.3'), fontsize=10)
plt.text(4.5, -2.0, f"GD Final:\n({final_gd[0]:.2f}, {final_gd[1]:.2f}, Z={int(final_gd_z)})",
         bbox=dict(facecolor='white', edgecolor='cyan', boxstyle='round,pad=0.3'), fontsize=10) """

# Plot settings
plt.xlabel('Intercept', fontsize=14)
plt.ylabel('Slope', fontsize=14)
plt.legend()
plt.title('Posterior Density with Paths of SGD and GD', fontsize=16)
plt.tight_layout()

# Adjust the axis limits to make space for text boxes
#plt.gca().set_xlim(-1, 1.5)  # Extend x-axis to include text boxes

# Save and show the plot
plt.savefig("PosteriorWithFinalPointsAndZ.png")
plt.show()
