from LoadData import X_data, y_data, sigma_likelihood, X_standardized, Y_standardized, sigma_likelihood_standardized
import torch
import numpy as np
from tqdm import tqdm
import random
from matplotlib import pyplot as plt
import math

def potential(z, X_data, y_data, sigma_prior, sigma_likelihood):
    # Prior term (Gaussian prior with variance sigma_prior^2)
    prior_theta_0 = 0.5 * (z[0] ** 2) / sigma_prior ** 2
    #prior_theta_0 = 0.5 * ((z[0] + 21.5) ** 2) / sigma_prior ** 2

    # Prior for slope (mean = 0.95)
    #prior_theta_1 = 0.5 * ((z[1] - 7.5) ** 2) / sigma_prior ** 2
    prior_theta_1 = 0.5 * ((z[1] - 0.955) ** 2) / sigma_prior ** 2

    prior = prior_theta_0 + prior_theta_1

    # Likelihood term (scaled by sigma_likelihood^2)
    intercept, slope = z[0], z[1]
    predicted_calories = intercept + slope * X_data
    residuals = y_data - predicted_calories
    likelihood = 0.5 * (residuals ** 2).sum() / sigma_likelihood ** 2

    # Negative log-posterior = prior + likelihood
    u = prior + likelihood
    return u


def stochastic_gradient_descent(X_data, y_data, sigma_likelihood, sigma_prior, n_samples=1e4, initial_step=1e-2, decay_rate=1e-5, alpha=0.55, batch_size=100):
    Z0 = torch.randn(2, requires_grad=True)  # 2 parameters: intercept and slope
    Zi = Z0
    trajectory = []  # To track the optimization progress

    for i in tqdm(range(int(n_samples))):
        # Sample a random mini-batch of indices
        indices = random.sample(range(len(X_data)), batch_size)
        X_batch = X_data[indices]
        y_batch = y_data[indices]

        N = len(X_data)
        n = len(X_batch)

        # Calculate step size with decay
        step = initial_step / (1 + decay_rate * i) ** alpha

        Zi.requires_grad_()
        u = potential(Zi, X_batch, y_batch, sigma_prior, sigma_likelihood)
        grad = (N / n) * torch.autograd.grad(u, Zi)[0]

        # Update parameters with gradient descent
        Zi = Zi.detach() - step * grad

        # Record the trajectory
        trajectory.append(Zi.detach().numpy())

    return np.array(trajectory)

def find_map_estimate(trajectory, X_data, y_data, sigma_prior, sigma_likelihood):

    if len(trajectory) == 0:
        raise ValueError("Trajectory is empty. Cannot find MAP estimate.")

    # Compute potentials for all parameter samples
    potentials = [
        potential(torch.tensor(z, requires_grad=False, dtype=torch.float32),
                  X_data, y_data, sigma_prior, sigma_likelihood).item()
        for z in trajectory
    ]

    # Find the index of the minimum potential
    min_index = np.argmin(potentials)
    return trajectory[min_index], potentials[min_index]

if __name__ == "__main__":

    trajectory = stochastic_gradient_descent(
    X_data=X_data,
    y_data=y_data,
    sigma_likelihood=sigma_likelihood,
    sigma_prior=1.0,
    n_samples=1e3,
    initial_step=1e-4, 
    decay_rate=1e-5, 
    alpha=0.55, 
    batch_size=100)

    # Scatter plot of samples
    plt.figure(figsize=(8, 6))
    plt.scatter(trajectory[:, 0], trajectory[:, 1], s=5, alpha=0.5, label="SGD Samples") 
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Scatter Plot of SGD Samples")
    plt.legend()
    plt.grid(True)
    plt.show()
