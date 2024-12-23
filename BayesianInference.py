import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# --- Define the prior distribution ---
np.random.seed(42)
prior_mean = np.array([0, 0])  # Mean of the prior (W$_0$, w1)
prior_cov = np.eye(2)  # Covariance matrix of the prior (std=1 for both)

x = np.linspace(prior_mean[0] - 2 * np.sqrt(prior_cov[0, 0]), prior_mean[0] + 2 * np.sqrt(prior_cov[0, 0]), 100)
y = np.linspace(prior_mean[1] - 2 * np.sqrt(prior_cov[1, 1]), prior_mean[1] + 2 * np.sqrt(prior_cov[1, 1]), 100)
plt.xlabel(r"$\theta_0$ (Intercept)", fontsize=14)
plt.ylabel(r"$\theta_1$ (Slope)", fontsize=14)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))
# Plot the contour of the prior distribution
rv = multivariate_normal(prior_mean, prior_cov)
Z_prior = rv.pdf(pos)

# Plotting the computed posterior density and sampled density on the same plot
fig, ax = plt.subplots(figsize=(8, 8))  # Create a single subplot

# Computed posterior density plot
contour_computed = ax.contourf(X, Y, Z_prior, 100, cmap='inferno')  # Computed posterior density as contour plot
fig.colorbar(contour_computed, ax=ax)
ax.set_xlabel(r"$\theta_0$ (Intercept)", fontsize=14)
ax.set_ylabel(r"$\theta_1$ (Slope)", fontsize=14)
fig.set_size_inches(6.4, 4.8)  # Full-width size
plt.savefig("PriorDensity.png")

# Sample 6 sets of (W$_0$, w1) from the prior
sampled_weights = np.random.multivariate_normal(prior_mean, prior_cov, size=6)

# --- Define the true function parameters ---
a_0, a_1 = 1, -0.5  # True parameters
x = np.linspace(-1, 1, 100)  # Input range for x
true_y = a_0 + a_1 * x  # True function

# --- Plot the true function and sampled lines ---
plt.figure(figsize=(8, 6))

# Plot the true function
#plt.plot(x, true_y, label="True Function (a_0 + a_1 * x)", color="black", linestyle="--")

# Plot the sampled lines
for i, (w0, w1) in enumerate(sampled_weights):
    sampled_y = w0 + w1 * x
    plt.plot(x, sampled_y, linestyle="-", alpha=0.8)

mean_intercept = np.mean(sampled_weights[:, 0])
mean_slope = np.mean(sampled_weights[:, 1])
std_intercept = np.std(sampled_weights[:, 0])
std_slope = np.std(sampled_weights[:, 1])
print(f"Mean Intercept Start(w$_0$): {mean_intercept},{std_intercept}")
print(f"Mean Slope Start(w1): {mean_slope},{std_slope}")


plt.xlabel("x", fontsize = 14)
plt.ylabel("y", fontsize = 14)
fig.set_size_inches(6.4, 4.8)  # Full-width size
plt.savefig("PriorSamples.png")
plt.xlim(-1,1)
plt.grid(False)
plt.show()

x_single = np.random.uniform(-1, 1)  # Random x in [-1, 1]
t_single = a_0 + a_1 * x_single + np.random.normal(0, 0.2)  # True label with Gaussian noise

# --- Plot the likelihood function p(t|x, w) ---
# Define the likelihood
def likelihood(t, x, w0, w1, sigma=0.2):
    """Likelihood function p(t|x, w)."""
    mean = w0 + w1 * x
    return np.exp(-0.5 * ((t - mean) / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)

# Compute the likelihood for the single data point
Z_likelihood = np.zeros_like(Z_prior)
for i in range(Z_likelihood.shape[0]):
    for j in range(Z_likelihood.shape[1]):
        Z_likelihood[i, j] = likelihood(t_single, x_single, X[i, j], Y[i, j])

        # Plot the likelihood

fig, ax = plt.subplots(figsize=(8, 8))
contour_likelihood = ax.contourf(X, Y, Z_likelihood, 100, cmap="inferno")
fig.colorbar(contour_likelihood, ax=ax)
ax.set_xlabel(r"$\theta_0$ (Intercept)", fontsize=14)
ax.set_ylabel(r"$\theta_1$ (Slope)", fontsize=14)
fig.set_size_inches(6.4, 4.8)  # Full-width size
plt.savefig("Likelihood1.png")
plt.show()

posterior = Z_prior * Z_likelihood  # Element-wise multiplication
posterior /= np.sum(posterior)

# --- Generate new samples from the updated posterior ---
# Approximate posterior sampling using the multivariate normal distribution
posterior_mean = [
    np.sum(X * posterior) / np.sum(posterior), 
    np.sum(Y * posterior) / np.sum(posterior)
]  # Weighted mean
posterior_cov = np.cov(
    X.flatten(), Y.flatten(), aweights=posterior.flatten()
)  # Weighted covariance matrix

# Ensure the covariance matrix is valid for sampling
posterior_cov = np.nan_to_num(posterior_cov, nan=1e-6)

# Sample 6 sets of (W$_0$, w1) from the posterior
sampled_weights_posterior = np.random.multivariate_normal(posterior_mean, posterior_cov, size=6)

# --- Plot the updated posterior samples ---
plt.figure(figsize=(8, 6))

# Plot the single data point
plt.scatter(x_single, t_single, color="blue", label="Single Data Point", zorder=14)

# Plot the sampled lines from the posterior
for i, (w0, w1) in enumerate(sampled_weights_posterior):
    y_sampled_posterior = w0 + w1 * x
    plt.plot(x, y_sampled_posterior, linestyle="-", alpha=0.8)

mean_intercept = np.mean(sampled_weights_posterior[:, 0])
mean_slope = np.mean(sampled_weights_posterior[:, 1])
std_intercept = np.std(sampled_weights_posterior[:, 0])
std_slope = np.std(sampled_weights_posterior[:, 1])
print(f"Mean Intercept Start(W$_0$): {mean_intercept},{std_intercept}")
print(f"Mean Slope Start(w1): {mean_slope},{std_slope}")


plt.xlabel("x", fontsize = 14)
plt.ylabel("y", fontsize = 14)
plt.xlim(-1, 1)
#fig.set_size_inches(6.4, 4.8)  # Full-width size
plt.savefig("PosteriorSamples1.png")
plt.legend()
plt.grid(False)
plt.show()

# --- Plot the posterior density ---
fig, ax = plt.subplots(figsize=(8, 8))  # Create a single subplot

# Computed posterior density plot
contour_computed = ax.contourf(X, Y, posterior, 100, cmap="inferno")  # Computed posterior density as contour plot
fig.colorbar(contour_computed, ax=ax)

ax.set_xlabel(r"$\theta_0$ (Intercept)", fontsize=14)
ax.set_ylabel(r"$\theta_1$ (Slope)", fontsize=14)
fig.set_size_inches(6.4, 4.8)  # Full-width size
plt.savefig("PosteriorDensity1.png")
plt.show()

# --- Draw another sample from the true function ---
x_double = np.random.uniform(-1, 1)  # Random x in [-1, 1]
t_double = a_0 + a_1 * x_double + np.random.normal(0, 0.2)  # True label with Gaussian noise

# --- Compute the likelihood for the new data point ---
Z_likelihood_2 = np.zeros_like(Z_prior)
for i in range(Z_likelihood_2.shape[0]):
    for j in range(Z_likelihood_2.shape[1]):
        Z_likelihood_2[i, j] = likelihood(t_double, x_double, X[i, j], Y[i, j])

# --- Plot the likelihood for the second data point ---
fig, ax = plt.subplots(figsize=(8, 8))
contour_likelihood_2 = ax.contourf(X, Y, Z_likelihood_2, 100, cmap="inferno")
fig.colorbar(contour_likelihood_2, ax=ax)
ax.set_xlabel(r"$\theta_0$ (Intercept)", fontsize=14)
ax.set_ylabel(r"$\theta_1$ (Slope)", fontsize=14)
fig.set_size_inches(6.4, 4.8)  # Full-width size
plt.savefig("Likelihood2.png")
plt.show()

# --- Update the posterior with the new likelihood ---
posterior_2 = posterior * Z_likelihood_2  # Element-wise multiplication
posterior_2 /= np.sum(posterior_2)  # Normalize

# --- Generate new samples from the updated posterior ---
posterior_mean_2 = [
    np.sum(X * posterior_2) / np.sum(posterior_2), 
    np.sum(Y * posterior_2) / np.sum(posterior_2)
]  # Weighted mean
posterior_cov_2 = np.cov(
    X.flatten(), Y.flatten(), aweights=posterior_2.flatten()
)  # Weighted covariance matrix

# Ensure the covariance matrix is valid for sampling
posterior_cov_2 = np.nan_to_num(posterior_cov_2, nan=1e-6)

# Sample 6 sets of (W$_0$, w1) from the updated posterior
sampled_weights_posterior_2 = np.random.multivariate_normal(posterior_mean_2, posterior_cov_2, size=6)

# --- Plot the updated posterior samples ---
plt.figure(figsize=(8, 6))

# Plot both data points
plt.scatter(x_single, t_single, color="blue", label="First Data Point", zorder=14)
plt.scatter(x_double, t_double, color="red", label="Second Data Point", zorder=14)

# Plot the sampled lines from the new posterior
for i, (w0, w1) in enumerate(sampled_weights_posterior_2):
    y_sampled_posterior_2 = w0 + w1 * x
    plt.plot(x, y_sampled_posterior_2, linestyle="-", alpha=0.8)

mean_intercept = np.mean(sampled_weights_posterior_2[:, 0])
mean_slope = np.mean(sampled_weights_posterior_2[:, 1])
std_intercept = np.std(sampled_weights_posterior_2[:, 0])
std_slope = np.std(sampled_weights_posterior_2[:, 1])
print(f"Mean Intercept Start(W$_0$): {mean_intercept},{std_intercept}")
print(f"Mean Slope Start(w1): {mean_slope},{std_slope}")



plt.xlabel("x",fontsize = 14)
plt.ylabel("y",fontsize = 14)
plt.xlim(-1, 1)
#fig.set_size_inches(6.4, 4.8)  # Full-width size
plt.legend()
plt.grid(False)
plt.savefig("PosteriorSamples2.png")   
plt.show()

# --- Plot the updated posterior density ---
fig, ax = plt.subplots(figsize=(8, 8))  # Create a single subplot

# Computed posterior density plot
contour_computed_2 = ax.contourf(X, Y, posterior_2, 100, cmap="inferno")  # Computed posterior density as contour plot
fig.colorbar(contour_computed_2, ax=ax)
ax.set_xlabel(r"$\theta_0$ (Intercept)", fontsize=14)
ax.set_ylabel(r"$\theta_1$ (Slope)", fontsize=14)
fig.set_size_inches(6.4, 4.8)  # Full-width size
plt.savefig("PosteriorDensity2.png")
plt.show()

# --- Draw 10 new samples iteratively and update the posterior ---
all_x_data = [x_single, x_double]  # Keep track of all data points
all_t_data = [t_single, t_double]
for _ in range(10):
    x_new = np.random.uniform(-1, 1)
    t_new = a_0 + a_1 * x_new + np.random.normal(0, 0.2)
    all_x_data.append(x_new)
    all_t_data.append(t_new)

    # Compute the likelihood for the current data point
    Z_likelihood_new = np.zeros_like(Z_prior)
    for i in range(Z_likelihood_new.shape[0]):
        for j in range(Z_likelihood_new.shape[1]):
            Z_likelihood_new[i, j] = likelihood(t_new, x_new, X[i, j], Y[i, j])
    
    # Update the posterior
    posterior = posterior * Z_likelihood_new
    posterior /= np.sum(posterior)

# Compute the final posterior mean and covariance
posterior_mean_final = [
    np.sum(X * posterior) / np.sum(posterior), 
    np.sum(Y * posterior) / np.sum(posterior)
]
posterior_cov_final = np.cov(
    X.flatten(), Y.flatten(), aweights=posterior.flatten()
)
posterior_cov_final = np.nan_to_num(posterior_cov_final, nan=1e-6)

# Sample from the final posterior
sampled_weights_posterior_final = np.random.multivariate_normal(
    posterior_mean_final, posterior_cov_final, size=6
)

# --- Plot the final likelihood for the last data point ---
fig, ax = plt.subplots(figsize=(8, 8))
contour_likelihood_final = ax.contourf(X, Y, Z_likelihood_new, 100, cmap="inferno")
fig.colorbar(contour_likelihood_final, ax=ax)
ax.set_xlabel(r"$\theta_0$ (Intercept)", fontsize=14)
ax.set_ylabel(r"$\theta_1$ (Slope)", fontsize=14)
fig.set_size_inches(6.4, 4.8)  # Full-width size
plt.savefig("LikelihoodFinal.png")
plt.show()

# --- Plot the final posterior samples ---
plt.figure(figsize=(8, 6))

# Plot all data points
plt.scatter(all_x_data, all_t_data, color="blue", label="All Data Points", zorder=12)

# Plot the sampled lines from the final posterior
# Plot the sampled lines from the final posterior
for i, (w0, w1) in enumerate(sampled_weights_posterior_final):
    plt.plot(x, w0 + w1 * x, linestyle="-", alpha=0.8)

mean_intercept = np.mean(sampled_weights_posterior_final[:, 0])
mean_slope = np.mean(sampled_weights_posterior_final[:, 1])
std_intercept = np.std(sampled_weights_posterior_final[:, 0])
std_slope = np.std(sampled_weights_posterior_final[:, 1])
print(f"Mean Intercept Start(W$_0$): {mean_intercept},{std_intercept}")
print(f"Mean Slope Start(w1): {mean_slope},{std_slope}")



plt.xlabel("x",fontsize = 14)
plt.ylabel("y",fontsize = 14)
plt.xlim(-1, 1)
plt.legend()
#fig.set_size_inches(6.4, 4.8)  # Full-width size
plt.grid(False)
plt.savefig("PosteriorSamplesFinal.png")
plt.show()

# --- Plot the final posterior density ---
fig, ax = plt.subplots(figsize=(8, 8))
contour_final = ax.contourf(X, Y, posterior, 100, cmap="inferno")
fig.colorbar(contour_final, ax=ax)
ax.set_xlabel(r"$\theta_0$ (Intercept)", fontsize=14)
ax.set_ylabel(r"$\theta_1$ (Slope)", fontsize=14)
fig.set_size_inches(6.4, 4.8)  # Full-width size
plt.savefig("PosteriorDensityFinal.png")
plt.show()