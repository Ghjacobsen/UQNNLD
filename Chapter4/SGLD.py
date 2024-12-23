from SGD import potential
import random
import torch
import numpy as np
from tqdm import tqdm
import math


def stochastic_gradient_langevin_dynamics(X_data, y_data, sigma_likelihood, sigma_prior, n_samples=1e4, initial_step=1e-6, decay_rate=1e-10, alpha=0.55, batch_size=100, map_estimate=None):
    burn_in = 0
    Z0 = torch.tensor(map_estimate, dtype=torch.float32) if map_estimate is not None else torch.randn(2, dtype=torch.float3)
    Zi = Z0
    samples = []

    for i in tqdm(range(int(n_samples + burn_in))):
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

        # Update with decaying step size and Gaussian noise
        noise_factor = 1
        Zi = Zi.detach() - step * grad + noise_factor * math.sqrt(2 * step) * torch.randn(2)

        if torch.isnan(Zi).any():
            print("NaN value encountered in Zi. Exiting program.")
            exit()

        if i >= burn_in:  # Collect samples after burn-in
            samples.append(Zi.detach().numpy())

    return np.array(samples)
