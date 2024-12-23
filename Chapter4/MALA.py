from SGD import potential
import random
import torch
import numpy as np
from tqdm import tqdm
import math

def log_Q(potential, z_prime, z, step, X_data, y_data, sigma_prior, sigma_likelihood):
    """
    Compute the log-density of the proposal q(Z_prime | Z).
    """
    z.requires_grad_()
    grad = torch.autograd.grad(potential(z, X_data, y_data, sigma_prior, sigma_likelihood).mean(), z)[0]
    # Log density of the proposal for MALA
    return -(torch.norm(z_prime - z + step * grad, p=2) ** 2) / (4 * step)

def metropolis_adjusted_langevin_algorithm(X_data, y_data, sigma_likelihood, sigma_prior, n_samples=100000, initial_step=1e-6, decay_rate=1e-10, alpha=0.55, map_estimate=None):
    acceptance_counter = 0
    total_counter = 0
    burn_in = 0  # Burn-in period to discard unrepresentative samples
    Z0 = torch.randn(2) if map_estimate is None else map_estimate
    Zi = Z0
    samples = []

    pbar = tqdm(range(int(n_samples + burn_in)))
    for i in pbar:
        Zi.requires_grad_()
        u = potential(Zi, X_data, y_data, sigma_prior, sigma_likelihood).mean()
        grad = torch.autograd.grad(u, Zi)[0]

        # Propose a new sample (MALA step)
        step = initial_step / (1 + decay_rate * i) ** alpha  # Step size with decay
        Z_next = Zi.detach() - step * grad + math.sqrt(2 * step) * torch.randn(2)  # Langevin proposal

        # Compute the acceptance probability
        Z_next.requires_grad_()
        u_next = potential(Z_next, X_data, y_data, sigma_prior, sigma_likelihood).mean()

        # Compute log-forward and log-reverse for proposal, using grad_next for Z_next
        log_forward = log_Q(potential, Z_next, Zi, step, X_data, y_data, sigma_prior, sigma_likelihood)
        log_reverse = log_Q(potential, Zi, Z_next, step, X_data, y_data, sigma_prior, sigma_likelihood)

        log_acceptance_ratio = -u_next + u + log_reverse - log_forward

        if torch.rand(1) < torch.exp(log_acceptance_ratio):
            Zi = Z_next
            acceptance_counter += 1
        samples.append(Zi.detach().numpy())
        total_counter += 1
        
    samples = np.array(samples)

    # Return the collected samples after burn-in
    print(f"Acceptance rate: {acceptance_counter / total_counter}")
    return samples[burn_in:]