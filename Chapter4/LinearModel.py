from LoadData import X_standardized, Y_standardized, sigma_likelihood_standardized, X_data, y_data, sigma_likelihood
from SGD import stochastic_gradient_descent, find_map_estimate
from Calcposterior import mu_posterior, Sigma_posterior,mu_posterior_standardized, Sigma_posterior_standardized
from SGLD import stochastic_gradient_langevin_dynamics
from MALA import metropolis_adjusted_langevin_algorithm
import seaborn as sns
from sklearn.model_selection import train_test_split
from scipy.stats import norm
import torch

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

trajectory = stochastic_gradient_descent(
    X_data=X_standardized,
    y_data=Y_standardized,
    sigma_likelihood=sigma_likelihood_standardized,
    sigma_prior=1.0,
    n_samples=1e4,
    initial_step=1e-6, 
    decay_rate=1e-5, 
    alpha=0.55, 
    batch_size=100)

map_estimate, min_potential = find_map_estimate(trajectory, 
                                                X_data = X_standardized, 
                                                y_data = Y_standardized, 
                                                sigma_prior = 1.0, 
                                                sigma_likelihood = sigma_likelihood_standardized)

print(f"MAP Estimate: {map_estimate}")
print(f"Minimum Potential: {min_potential}")

SGDSlope = map_estimate[1]
SGDIntercept = map_estimate[0]


posterior_samples_SGLD = stochastic_gradient_langevin_dynamics(
    X_data=X_standardized,
    y_data=Y_standardized,
    sigma_likelihood=sigma_likelihood_standardized,
    sigma_prior=1.0,
    n_samples=5e4,
    initial_step=1e-5,
    decay_rate=1e-3,
    alpha=0.55,
    batch_size=100,
    map_estimate=np.random.rand(2)
    ) 
slope_samples = posterior_samples_SGLD[:, 1]
intercept_samples = posterior_samples_SGLD[:, 0]
print(f"{slope_samples.mean(), slope_samples.std(), intercept_samples.mean(), intercept_samples.std()}")


"""
#map_estimate = torch.randn(2)  # Initial map estimate (random in this case)
posterior_samples_SGLD_Mala = metropolis_adjusted_langevin_algorithm(
    X_data=X_standardized,
    y_data=Y_standardized,
    sigma_likelihood=sigma_likelihood_standardized,
    sigma_prior=1.0,
    n_samples=5e4,
    initial_step=1e-5,
    decay_rate=1e-3,
    alpha=0.55,
    map_estimate=torch.randn(2),
)

slope_samples_Mala = posterior_samples_SGLD_Mala[:, 1]
intercept_samples_Mala = posterior_samples_SGLD_Mala[:, 0]

print(f"{slope_samples_Mala.mean(), slope_samples_Mala.std(), intercept_samples_Mala.mean(), intercept_samples_Mala.std()}")
"""
 

def plot_samples_countour(samples, mu_posterior, Sigma_posterior, scaling=20, figure_name="plot.pdf"):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import multivariate_normal

    # Define the grid for the contour plot
    x = np.linspace(
        mu_posterior[0] - scaling * np.sqrt(Sigma_posterior[0, 0]),
        mu_posterior[0] + scaling * np.sqrt(Sigma_posterior[0, 0]),
        100
    )
    y = np.linspace(
        mu_posterior[1] - scaling * np.sqrt(Sigma_posterior[1, 1]),
        mu_posterior[1] + scaling * np.sqrt(Sigma_posterior[1, 1]),
        100
    )
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))

    # Compute the multivariate normal density
    rv = multivariate_normal(mu_posterior, Sigma_posterior)
    Z_computed = rv.pdf(pos)

    # Plotting
    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    # Computed posterior density as contour plot
    contour_computed = ax.contourf(X, Y, Z_computed, levels=100, cmap='inferno')
    fig.colorbar(contour_computed, ax=ax, label='Density')

    # Scatter plot of samples
    ax.scatter(
        samples[:, 1], samples[:, 0],  # Assuming samples are [n_samples, 2]
        color='white', s=10, alpha=0.6, label="Scatter Samples"
    )

    # Set axis limits to match the computed posterior density plot
    ax.set_xlim(
        mu_posterior[0] - scaling * np.sqrt(Sigma_posterior[0, 0]),
        mu_posterior[0] + scaling * np.sqrt(Sigma_posterior[0, 0])
    )
    ax.set_ylim(
        mu_posterior[1] - scaling * np.sqrt(Sigma_posterior[1, 1]),
        mu_posterior[1] + scaling * np.sqrt(Sigma_posterior[1, 1])
    )

    # Labels and legend
    plt.ylabel('Intercept', fontsize=14)
    plt.xlabel('Slope', fontsize=14)
    plt.legend()
    plt.tight_layout()

    plt.savefig(figure_name)
    plt.show()
    plt.close()



"""
plt.figure(figsize=(12.8, 9.6))
 # Trace plot for slope
plt.subplot(1, 2, 1)  # Plot on the left side of a 1x2 grid
plt.plot(slope_samples, color='b', label='Slope Samples')
plt.title('Trace Plot for Slope')
plt.xlabel('Iteration')
plt.ylabel('Slope Value')
plt.grid(True)

# Trace plot for intercept
plt.subplot(1, 2, 2)  # Plot on the right side of a 1x2 grid
plt.plot(intercept_samples, color='r', label='Intercept Samples')
plt.title('Trace Plot for Intercept')
plt.xlabel('Iteration')
plt.ylabel('Intercept Value')
plt.grid(True)

# Display the plots
plt.tight_layout()
plt.savefig("SGLDTracePlots.png")
plt.show() 

"""
plt.figure(figsize=(12.8, 9.6))

# Trace plot for slope
plt.subplot(1, 2, 1)  # Plot on the left side of a 1x2 grid
plt.plot(slope_samples_Mala, color='b', label='Slope Samples')
plt.title('Trace Plot for Slope')
plt.xlabel('Iteration')
plt.ylabel('Slope Value')
plt.grid(True)

# Trace plot for intercept
plt.subplot(1, 2, 2)  # Plot on the right side of a 1x2 grid
plt.plot(intercept_samples_Mala, color='r', label='Intercept Samples')
plt.title('Trace Plot for Intercept')
plt.xlabel('Iteration')
plt.ylabel('Intercept Value')
plt.grid(True)

# Display the plots
plt.tight_layout()
plt.savefig("MALATracePlots.pdf")
plt.show()

#plot_samples_countour(trajectory, mu_posterior_standardized, Sigma_posterior_standardized, scaling=10, figure_name="SGDPathLinearModel.png")
#plot_samples_countour(posterior_samples_SGLD, mu_posterior_standardized, Sigma_posterior_standardized, scaling=15, figure_name="SGLDLinearModel.png")
plot_samples_countour(posterior_samples_SGLD_Mala, mu_posterior_standardized, Sigma_posterior_standardized, scaling=15, figure_name="MALALinearModel.pdf")
#plot_samples_countour(posterior_samples_SGLD_Mala, mu_posterior_standardized, Sigma_posterior_standardized)
#plot_samples_countour(trajectory, mu_posterior_standardized, Sigma_posterior_standardized)

slope_range = np.linspace(
    mu_posterior_standardized[0] - 3 * np.sqrt(Sigma_posterior_standardized[0, 0]),
    mu_posterior_standardized[0] + 3 * np.sqrt(Sigma_posterior_standardized[0, 0]),
    100
)
intercept_range = np.linspace(
    mu_posterior_standardized[1] - 3 * np.sqrt(Sigma_posterior_standardized[1, 1]),
    mu_posterior_standardized[1] + 3 * np.sqrt(Sigma_posterior_standardized[1, 1]),
    100
)

# Compute Gaussian density for intercept
intercept_density = norm.pdf(
    intercept_range,
    loc=mu_posterior_standardized[1],
    scale=np.sqrt(Sigma_posterior_standardized[1,1])
)

# Compute Gaussian density for slope
slope_density = norm.pdf(
    slope_range,
    loc=mu_posterior_standardized[0],
    scale=np.sqrt(Sigma_posterior_standardized[0, 0])
)

plt.figure(figsize=(6.4, 4.8))
sns.kdeplot(slope_samples_Mala, fill=True, color="blue", alpha=1, label="Slope Samples")
plt.plot(slope_range, slope_density, color="lightblue", label="Slope True Values")
plt.fill_between(slope_range, slope_density, alpha=0.6, color="lightblue")
plt.axvline(SGDSlope, color="black", linestyle="--", linewidth=2, alpha = 0.3, label="SGD MAP Estimate")
plt.xlabel("Slope", fontsize=14)
plt.ylabel("Density", fontsize=14)
plt.xlim(0.9, 1.0)
plt.legend()
plt.tight_layout()
plt.savefig("MALALinModelSlopeDensity.pdf")
plt.show()

plt.figure(figsize=(6.4, 4.8))
sns.kdeplot(intercept_samples_Mala, fill=True, color="green", alpha=1, label="Intercept Samples")
plt.plot(intercept_range, intercept_density, color="lightgreen", label="Intercept True Values")
plt.fill_between(intercept_range, intercept_density, alpha=0.6, color="lightgreen")
plt.axvline(SGDIntercept, color="black", linestyle="--", linewidth=2, alpha = 0.3, label="SGD MAP Estimate")
plt.xlabel("Intercept", fontsize=14)
plt.ylabel("Density", fontsize=14)
plt.legend()
plt.xlim(-0.04, 0.04)
plt.tight_layout()
plt.savefig("MALALinModelInterceptDensity.pdf")
plt.show()
