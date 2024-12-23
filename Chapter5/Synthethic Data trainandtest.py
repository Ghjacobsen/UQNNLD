from sklearn.model_selection import train_test_split
from Classes import Langevin_Wrapper, SGD_Wrapper
from sklearn.metrics import mean_squared_error
import numpy as np
import GPy
import copy
import matplotlib.pyplot as plt
import torch
from scipy.stats import norm

# Data generation
np.random.seed(2)
no_points = 400
lengthscale = 1
variance = 1.0
sig_noise = 0.3
x = np.random.uniform(-3, 3, no_points)[:, None]
x.sort(axis=0)

k = GPy.kern.RBF(input_dim=1, variance=variance, lengthscale=lengthscale)
C = k.K(x, x) + np.eye(no_points) * sig_noise**2

y = np.random.multivariate_normal(np.zeros((no_points)), C)[:, None]
y = (y - y.mean())


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) 


# Convert to tensors for training/testing
x_train_tensor = torch.tensor(x_train, dtype=torch.float32).cpu()
x_test_tensor = torch.tensor(x_test, dtype=torch.float32).cpu()
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).cpu()
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).cpu()


net = Langevin_Wrapper(input_dim=1, output_dim=1, no_units=200, learn_rate=1e-5, init_log_noise=0, weight_decay=0.1)
# SGD Wrapper setup
sgd_net = SGD_Wrapper(input_dim=1, output_dim=1, no_units=200, learn_rate=1e-4,init_log_noise = 0, weight_decay=0.1)


num_epochs = 2000
# Train Langevin model (SGLD)
nets = []
mix_epochs, burnin_epochs = 100, 1000
batch_size = len(x_train) // 10  # Mini-batch size

for epoch in range(num_epochs):
    # Shuffle data for mini-batch training
    perm = np.random.permutation(len(x_train))
    x_train_shuffled = x_train[perm]
    y_train_shuffled = y_train[perm]

    for i in range(0, len(x_train), batch_size):
        x_batch = x_train_shuffled[i:i + batch_size]
        y_batch = y_train_shuffled[i:i + batch_size]

        # Convert mini-batches to tensors
        x_batch_tensor = torch.tensor(x_batch, dtype=torch.float32).cpu()
        y_batch_tensor = torch.tensor(y_batch, dtype=torch.float32).cpu()

        # Perform one SGLD step on the mini-batch
        loss = net.fit(x_batch_tensor, y_batch_tensor)

    # Logging and posterior sampling
    if epoch % 200 == 0:
        print(f'SGLD Epoch: {epoch}, Train Loss = {loss.cpu().data.numpy():.3f}')
    if epoch % mix_epochs == 0 and epoch > burnin_epochs:
        nets.append(copy.deepcopy(net.network))

# Train SGD model
for epoch in range(num_epochs):
    # Shuffle data for mini-batch training
    perm = np.random.permutation(len(x_train))
    x_train_shuffled = x_train[perm]
    y_train_shuffled = y_train[perm]

    for i in range(0, len(x_train), batch_size):
        x_batch = x_train_shuffled[i:i + batch_size]
        y_batch = y_train_shuffled[i:i + batch_size]

        # Convert mini-batches to tensors
        x_batch_tensor = torch.tensor(x_batch, dtype=torch.float32).cpu()
        y_batch_tensor = torch.tensor(y_batch, dtype=torch.float32).cpu()

        # Perform one SGD step on the mini-batch
        loss = sgd_net.fit(x_batch_tensor, y_batch_tensor)

    # Logging
    if epoch % 200 == 0:
        print(f'SGD Epoch: {epoch}, Train Loss = {loss.cpu().data.numpy():.3f}')



# SGLD Predictions
samples = []
noises = []
for network in nets:
    preds = network.forward(x_test_tensor).cpu().data.numpy()  # Test predictions
    samples.append(preds)
    noises.append(torch.exp(network.log_noise).cpu().data.numpy())

samples = np.array(samples)
noises = np.array(noises).reshape(-1)
means = samples.mean(axis=0).flatten()

# SGD Predictions
sgd_preds_test = sgd_net.network(x_test_tensor).cpu().data.numpy()

# Evaluate RMSE and NLL for SGLD
rmses = []
log_likelihoods = []
sigma_likelihood = noises.mean()
epsilon = 1e-6

for sample_preds in samples:
    rmse = np.sqrt(mean_squared_error(y_test.flatten(), sample_preds.flatten()))
    rmses.append(rmse)
    log_likelihood = np.mean(norm.logpdf(y_test.flatten(), loc=sample_preds.flatten(), scale=np.sqrt(np.clip(sigma_likelihood, epsilon, None)**2)))
    log_likelihoods.append(log_likelihood)

mean_rmse = np.mean(rmses)
std_rmse = np.std(rmses)
mean_log_likelihood = np.mean(log_likelihoods)
std_log_likelihood = np.std(log_likelihoods)

# Evaluate RMSE and NLL for SGD
sgd_rmse = np.sqrt(mean_squared_error(y_test.flatten(), sgd_preds_test.flatten()))
sgd_log_likelihood = np.mean(norm.logpdf(y_test.flatten(), loc=sgd_preds_test.flatten(), scale=sigma_likelihood))

# Print evaluation results
print(f"SGD RMSE (Test Data): {sgd_rmse:.4f}")
print(f"SGLD Mean RMSE: {mean_rmse:.4f}")
print(f"SGLD RMSE Uncertainty (Standard Deviation): {std_rmse:.4f}\n")
print(f"SGD Log-Likelihood (Test Data): {sgd_log_likelihood:.4f}")
print(f"SGLD Mean Log-Likelihood: {mean_log_likelihood:.4f}")
print(f"SGLD Log-Likelihood Uncertainty (Standard Deviation): {std_log_likelihood:.4f}")

# Plot predictions
c = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
# Compute aleatoric and epistemic uncertainties for the training interval
aleatoric = noises.mean()
epistemic = (samples.var(axis=0) ** 0.5).reshape(-1)
total_unc = (aleatoric**2 + epistemic**2)**0.5

# Extend x beyond the training interval
x_extended = np.linspace(-5, 5, int(no_points * 1.5))[:, None]  # More points for smoother visualization
x_extended_tensor = torch.tensor(x_extended, dtype=torch.float32).cpu()

# Collect predictions for extended range
samples_extended = []
noises_extended = []
for network in nets:
    preds = network.forward(x_extended_tensor).cpu().data.numpy()  # Predictions for the extended range
    samples_extended.append(preds)
    noises_extended.append(torch.exp(network.log_noise).cpu().data.numpy())  # Extract log noise

samples_extended = np.array(samples_extended)
noises_extended = np.array(noises_extended).reshape(-1)

# Compute mean and uncertainties for the extended interval
means_extended = samples_extended.mean(axis=0).reshape(-1)
sgd_preds_extended = sgd_net.network(x_extended_tensor).cpu().data.numpy()
aleatoric_extended = noises_extended.mean()
epistemic_extended = (samples_extended.var(axis=0) ** 0.5).reshape(-1)
total_unc_extended = (aleatoric_extended**2 + epistemic_extended**2)**0.5

# Fit a Gaussian Process Regression (GPR) model
gpr_model = GPy.models.GPRegression(x_train, y_train, kernel=k)

# Optimize the GPR model to find the best hyperparameters
gpr_model.optimize(messages=False)

# Predict the mean and variance of the GPR model for the extended range
mean_extended, var_extended = gpr_model.predict(x_extended)

# ------------------------------------------------------------
# Figure 1: SGLD Plot with Uncertainty
plt.figure(figsize=(6.4, 4.8))
plt.style.use('default')

# Plot training data
plt.scatter(x_train, y_train, s=10, marker='x', color=c[7], alpha=0.5, label='Training data')

# Plot extended uncertainty regions (SGLD)
plt.fill_between(
    x_extended.flatten(),
    (means_extended + aleatoric_extended),
    (means_extended + total_unc_extended),
    color=c[0],
    alpha=0.3,
    label='Epistemic + Aleatoric'
)
plt.fill_between(
    x_extended.flatten(),
    (means_extended - total_unc_extended),
    (means_extended - aleatoric_extended),
    color=c[0],
    alpha=0.3
)
plt.fill_between(
    x_extended.flatten(),
    (means_extended - aleatoric_extended),
    (means_extended + aleatoric_extended),
    color=c[1],
    alpha=0.4,
    label='Aleatoric'
)

# Plot mean predictions
plt.plot(x_extended.flatten(), means_extended, color='black', linewidth=2, label='SGLD Mean Prediction')

# Configure plot
plt.xlim([-5, 5])
plt.ylim([-5, 7])
plt.xlabel('$x$', fontsize=30)
plt.title('SGLD', fontsize=40)
plt.tick_params(labelsize=30)
plt.xticks(np.arange(-5, 6, 2))
plt.yticks(np.arange(-4, 7, 2))
plt.gca().set_yticklabels([])
plt.gca().yaxis.grid(alpha=0.3)
plt.gca().xaxis.grid(alpha=0.3)
plt.legend(fontsize=8, loc='upper right')
plt.tight_layout()
plt.savefig("SGLDBaselineNN2.png")
plt.show()

# ------------------------------------------------------------
# Figure 2: SGD Plot
plt.figure(figsize=(6.4, 4.8))
plt.style.use('default')

# Plot training data
plt.scatter(x_train, y_train, s=10, marker='x', color=c[7], alpha=0.5, label='Training data')

# Plot SGD baseline predictions
plt.plot(x_extended.flatten(), sgd_preds_extended.flatten(), color='black', linewidth=2, label='SGD Baseline')

# Configure plot
plt.xlim([-5, 5])
plt.ylim([-5, 7])
plt.xlabel('$x$', fontsize=30)
plt.title('SGD', fontsize=40)
plt.tick_params(labelsize=30)
plt.xticks(np.arange(-5, 6, 2))
plt.yticks(np.arange(-4, 7, 2))
plt.gca().set_yticklabels([])
plt.gca().yaxis.grid(alpha=0.3)
plt.gca().xaxis.grid(alpha=0.3)
plt.legend(fontsize=8, loc='upper right')
plt.tight_layout()
plt.savefig("SGDBaselineNN2.png")
plt.show()

# Compute uncertainties (Epistemic + Aleatoric)
aleatoric_extended = noises_extended.mean()  # Aleatoric uncertainty
epistemic_extended = np.sqrt(var_extended).flatten()  # Epistemic uncertainty from GPR variance
total_unc_extended = np.sqrt(aleatoric_extended**2 + epistemic_extended**2)  # Total uncertainty

# Plot the ground truth function with total uncertainty
plt.figure(figsize=(6.4, 4.8))
plt.style.use('default')

# Plot training data
plt.scatter(x_train, y_train, s=10, marker='x', color=c[7], alpha=0.5, label='Training Data')

# Plot the ground truth function (mean of GPR)
plt.plot(x_extended, mean_extended, color='black', linewidth=2, label='Ground Truth Function')

# Plot extended uncertainty regions (Epistemic + Aleatoric)
plt.fill_between(
    x_extended.flatten(),
    (mean_extended.flatten() + aleatoric_extended),
    (mean_extended.flatten() + total_unc_extended),
    color=c[0],
    alpha=0.3,
    label='Epistemic + Aleatoric'
)
plt.fill_between(
    x_extended.flatten(),
    (mean_extended.flatten() - total_unc_extended),
    (mean_extended.flatten() - aleatoric_extended),
    color=c[0],
    alpha=0.3
)
plt.fill_between(
    x_extended.flatten(),
    (mean_extended.flatten() - aleatoric_extended),
    (mean_extended.flatten() + aleatoric_extended),
    color=c[1],
    alpha=0.4,
    label='Aleatoric'
)

# Configure plot
plt.xlim([-5, 5])
plt.ylim([-5, 7])
plt.xlabel('$x$', fontsize=30)
plt.title('GPR', fontsize=40)
plt.tick_params(labelsize=30)
plt.xticks(np.arange(-5, 6, 2))
plt.yticks(np.arange(-4, 7, 2))
plt.gca().set_yticklabels([])
plt.gca().yaxis.grid(alpha=0.3)
plt.gca().xaxis.grid(alpha=0.3)
plt.legend(fontsize=8, loc='upper right')
plt.tight_layout()
plt.savefig("GPR2.png")
plt.show()