from LoadData import X_data, y_data, sigma_likelihood, X_standardized, Y_standardized, sigma_likelihood_standardized
from scipy.stats import multivariate_normal
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.stats import norm
sys.stdout.reconfigure(encoding='utf-8')

# Define the prior (normal distribution for slope and intercept)
def log_prior(theta):
    mu_prior = torch.tensor([0.0, 0.0])
    sigma_prior = 1.0
    return -0.5 * torch.sum((theta - mu_prior) ** 2) / sigma_prior ** 2

# Define the likelihood function
def log_likelihood(theta, X_data, y_data, sigma=3.0):
    slope, intercept = theta
    y_pred = slope * X_data + intercept
    return -0.5 * torch.sum((y_data - y_pred) ** 2) / sigma ** 2 - 0.5 * len(y_data) * torch.log(torch.tensor(2 * np.pi * sigma ** 2))

# Define the unnormalized posterior
def log_posterior(theta, X_data, y_data):
    return log_prior(theta) + log_likelihood(theta, X_data, y_data)

def Calculate_Posterior(X_data, y_data, sigma_likelihood, sigma_prior):
    """
    Calculate the posterior mean and covariance matrix for Bayesian linear regression.

    Parameters:
    X_data : torch.Tensor
        Feature data (1D tensor of shape [n_samples]).
    y_data : torch.Tensor
        Target data (1D tensor of shape [n_samples]).
    sigma_likelihood : float
        Standard deviation of the likelihood.
    sigma_prior : float
        Standard deviation of the prior.

    Returns:
    mu_posterior : numpy.ndarray
        Posterior mean vector (1D array).
    Sigma_posterior : numpy.ndarray
        Posterior covariance matrix (2D array).
    """
    # Prepare the design matrix
    X_design = torch.stack([X_data, torch.ones_like(X_data)], dim=1)  # Add intercept as a column

    # Identity matrix for intercept and slope dimensions
    I = torch.eye(2)

    # Compute squared variances
    sigma_prior_sq = sigma_prior ** 2
    sigma_likelihood_sq = sigma_likelihood ** 2

    # Regularization value to prevent singular matrix issues
    regularization = 1e-6

    # Posterior covariance matrix: Σ_posterior = (X^T X / σ_likelihood^2 + I / σ_prior^2)^{-1}
    XT_X = X_design.T @ X_design
    regularized_matrix = XT_X / sigma_likelihood_sq + I / sigma_prior_sq + regularization * torch.eye(I.shape[0])
    Sigma_posterior = torch.inverse(regularized_matrix)

    # Posterior mean: μ_posterior = Σ_posterior * (X^T y / σ_likelihood^2)
    XT_y = X_design.T @ y_data
    mu_posterior = Sigma_posterior @ (XT_y / sigma_likelihood_sq)

    # Convert results to numpy for sampling
    mu_posterior = mu_posterior.numpy()
    Sigma_posterior = Sigma_posterior.numpy()

    return mu_posterior, Sigma_posterior

mu_posterior, Sigma_posterior = Calculate_Posterior(X_data, y_data, sigma_likelihood, 10.0)
mu_posterior_standardized, Sigma_posterior_standardized = Calculate_Posterior(X_standardized, Y_standardized, sigma_likelihood_standardized, 1.0)

# Print results
#print("Posterior Mean (mu):", mu_posterior)
#print("Posterior Covariance (sigma):", Sigma_posterior)

if __name__== "__main__":
    #STANDARDIZED
    x = np.linspace(mu_posterior_standardized[0] - 3 * np.sqrt(Sigma_posterior_standardized[0, 0]), mu_posterior_standardized[0] + 3 * np.sqrt(Sigma_posterior_standardized[0, 0]), 100)
    y = np.linspace(mu_posterior_standardized[1] - 3 * np.sqrt(Sigma_posterior_standardized[1, 1]), mu_posterior_standardized[1] + 3 * np.sqrt(Sigma_posterior_standardized[1, 1]), 100)
    #print(samples_SGLD[0])
    #x= np.linspace(-32, -12, 100)
    #y = np.linspace(6,8,100)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))

    # Compute the multivariate normal density
    rv = multivariate_normal(mu_posterior_standardized, Sigma_posterior_standardized)
    Z_computed = rv.pdf(pos)

    # Plotting the computed posterior density and sampled density on the same plot
    fig, ax = plt.subplots(figsize=(6.4, 4.8))  # Create a single subplot

    # Computed posterior density plot
    contour_computed = ax.contourf(X, Y, Z_computed, 100, cmap='inferno')  # Computed posterior density as contour plot
    fig.colorbar(contour_computed, ax=ax, label='Density')

    # Set x and y limits to match the computed posterior density plot
    ax.set_xlim(mu_posterior_standardized[0] - 3 * np.sqrt(Sigma_posterior_standardized[0, 0]), mu_posterior_standardized[0] + 3 * np.sqrt(Sigma_posterior_standardized[0, 0]))
    ax.set_ylim(mu_posterior_standardized[1] - 3 * np.sqrt(Sigma_posterior_standardized[1, 1]), mu_posterior_standardized[1] + 3 * np.sqrt(Sigma_posterior_standardized[1, 1]))
    # Adjust layout and show the plot
    plt.ylabel('Intercept',fontsize = 14)
    plt.xlabel('Slope',fontsize = 14)
    plt.savefig("StandardizedPOSTERIOR.png")
    plt.tight_layout()
    plt.show() 

    #NOT STANDARDIZED
    x = np.linspace(mu_posterior[0] - 3 * np.sqrt(Sigma_posterior[0, 0]), mu_posterior[0] + 3 * np.sqrt(Sigma_posterior[0, 0]), 100)
    y = np.linspace(mu_posterior[1] - 3 * np.sqrt(Sigma_posterior[1, 1]), mu_posterior[1] + 3 * np.sqrt(Sigma_posterior[1, 1]), 100)
    #print(samples_SGLD[0])
    #x= np.linspace(-32, -12, 100)
    #y = np.linspace(6,8,100)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))

    # Compute the multivariate normal density
    rv = multivariate_normal(mu_posterior, Sigma_posterior)
    Z_computed = rv.pdf(pos)

    # Plotting the computed posterior density and sampled density on the same plot
    fig, ax = plt.subplots(figsize=(6.4, 4.8))  # Create a single subplot

    # Computed posterior density plot
    contour_computed = ax.contourf(X, Y, Z_computed, 100, cmap='inferno')  # Computed posterior density as contour plot
    fig.colorbar(contour_computed, ax=ax, label='Density')

    # Set x and y limits to match the computed posterior density plot
    ax.set_xlim(mu_posterior[0] - 3 * np.sqrt(Sigma_posterior[0, 0]), mu_posterior[0] + 3 * np.sqrt(Sigma_posterior[0, 0]))
    ax.set_ylim(mu_posterior[1] - 3 * np.sqrt(Sigma_posterior[1, 1]), mu_posterior[1] + 3 * np.sqrt(Sigma_posterior[1, 1]))
    # Adjust layout and show the plot
    plt.ylabel('Intercept',fontsize = 14)
    plt.xlabel('Slope',fontsize = 14)
    plt.savefig("PosteriorDensityForLinMOdel.png")
    plt.tight_layout()
    plt.show() 


if __name__ == "__main__":
    # Define range for intercept and slope based on analytical posterior
    intercept_range = np.linspace(
        mu_posterior_standardized[0] - 3 * np.sqrt(Sigma_posterior_standardized[0, 0]),
        mu_posterior_standardized[0] + 3 * np.sqrt(Sigma_posterior_standardized[0, 0]),
        100
    )
    slope_range = np.linspace(
        mu_posterior_standardized[1] - 3 * np.sqrt(Sigma_posterior_standardized[1, 1]),
        mu_posterior_standardized[1] + 3 * np.sqrt(Sigma_posterior_standardized[1, 1]),
        100
    )
    print(f"{mu_posterior_standardized[0], mu_posterior_standardized[1], np.sqrt(Sigma_posterior_standardized[0, 0]), np.sqrt(Sigma_posterior_standardized[1, 1])}")
    print(f"{mu_posterior[0], mu_posterior[1], np.sqrt(Sigma_posterior[0, 0]), np.sqrt(Sigma_posterior[1, 1])}")


    # Compute Gaussian density for intercept
    intercept_density = norm.pdf(
        intercept_range,
        loc=mu_posterior_standardized[0],
        scale=np.sqrt(Sigma_posterior_standardized[0, 0])
    )
    
    # Plot histogram for intercept
    plt.figure(figsize=(6.4, 4.8))
    plt.plot(intercept_range, intercept_density, color="blue", label="Slope Density")
    plt.fill_between(intercept_range, intercept_density, alpha=0.5, color="blue")
    plt.xlabel("Slope", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.legend()
    plt.tight_layout()
    #plt.savefig("InterceptDensity.png")
    plt.show()

    # Compute Gaussian density for slope
    slope_density = norm.pdf(
        slope_range,
        loc=mu_posterior_standardized[1],
        scale=np.sqrt(Sigma_posterior_standardized[1, 1])
    )

    # Plot histogram for slope
    plt.figure(figsize=(6.4, 4.8))
    plt.plot(slope_range, slope_density, color="green", label="Intercept Density")
    plt.fill_between(slope_range, slope_density, alpha=0.5, color="green")
    plt.xlabel("Intercept", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig("SlopeDensity.png")
    plt.show()
