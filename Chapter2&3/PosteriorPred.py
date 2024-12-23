import numpy as np
import matplotlib.pyplot as plt

# --- True function parameters ---
a_0, a_1 = 1, -0.5  # True parameters

# --- Data Generation ---
np.random.seed(42)
x_data = np.random.uniform(-1, 1, 50)  # Generate 20 random x values
t_data = a_0 + a_1 * x_data + np.random.normal(0, 0.2, size=len(x_data))  # Generate noisy target values

# --- Define Prior Distribution ---
prior_mean = np.array([0, 0])  # Mean of the prior (w0, w1)
prior_cov = np.eye(2)  # Covariance of the prior (identity matrix)
sigma_likelihood = 0.2  # Standard deviation of noise

x_range = np.linspace(-1, 1, 100)  # Continuous x-range for plotting
true_y = a_0 + a_1 * x_range  # True function

# --- Iteratively Update Posterior and Plot ---
for n in range(1, len(x_data) + 1):
    # Use only the first n data points
    x_subset = x_data[:n]
    t_subset = t_data[:n]

    # Design matrix
    X = np.vstack([np.ones(len(x_subset)), x_subset]).T

    # Update posterior mean and covariance
    precision_prior = np.linalg.inv(prior_cov)
    precision_likelihood = (X.T @ X) / sigma_likelihood**2
    precision_posterior = precision_prior + precision_likelihood
    posterior_cov = np.linalg.inv(precision_posterior)

    posterior_mean = posterior_cov @ (precision_prior @ prior_mean + X.T @ t_subset / sigma_likelihood**2)

    # Update prior with the current posterior
    prior_mean = posterior_mean  # Set the posterior mean as the new prior mean
    prior_cov = posterior_cov    # Set the posterior covariance as the new prior covariance

    # Sample 6 weights from the posterior
    sampled_weights = np.random.multivariate_normal(posterior_mean, posterior_cov, size=6)

    # Only plot for N=1, N=2, N=5, and N=20
    if n not in [1, 2, 5, 50]:
        continue

    # --- Plotting ---
    plt.figure(figsize=(6.4, 4.8))
    

    # Plot true function
    plt.plot(x_range, true_y, label="True Function", color="black", linestyle="--")

    # Plot data points used so far
    plt.scatter(x_subset, t_subset, color="blue", label=f"Data Points (n={n})", zorder=10)

    # Plot mean function
    mean_y = posterior_mean[0] + posterior_mean[1] * x_range
    plt.plot(x_range, mean_y, color="red", label="Predictive Mean", linewidth=2)

    # Plot uncertainty region
    y_uncertainty = np.sqrt(posterior_cov[0, 0] + posterior_cov[1, 1] * x_range**2)
    plt.fill_between(
        x_range,
        mean_y - 2 * y_uncertainty,
        mean_y + 2 * y_uncertainty,
        color="red",
        alpha=0.2,
        #label="95% CI"
    )

    # Labels and Legend
    plt.xlabel("x", fontsize=14)
    plt.ylabel("y", fontsize=14)
    
    plt.title(f"Posterior Updates with n={n} Data Points", fontsize = 16)
    plt.legend()
    plt.grid(False)
    plt.xlim(-1, 1)
    plt.ylim(min(t_data) - 1, max(t_data) + 1)
    plt.savefig(f"PredictiveUpdate_n{n}.png")
    plt.show()
