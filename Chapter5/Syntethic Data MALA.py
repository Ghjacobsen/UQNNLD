from Classes import Langevin_Wrapper, SGD_Wrapper, MALA_Wrapper
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import GPy
import copy
import matplotlib.pyplot as plt
import torch
from scipy.stats import norm

np.random.seed(2)
no_points = 400
lengthscale = 1
variance = 1.0
sig_noise = 0.3
x = np.random.uniform(-3, 3, no_points)[:, None]
x.sort(axis = 0)

k = GPy.kern.RBF(input_dim = 1, variance = variance, lengthscale = lengthscale)
C = k.K(x, x) + np.eye(no_points)*sig_noise**2

y = np.random.multivariate_normal(np.zeros((no_points)), C)[:, None]
y = (y - y.mean())
#Changing values for interval of parameters to train on. 



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Convert to tensors for training/testing
x_train_tensor = torch.tensor(x_train, dtype=torch.float32).cpu()
x_test_tensor = torch.tensor(x_test, dtype=torch.float32).cpu()
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).cpu()
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).cpu()

best_net, best_loss = None, float('inf')
num_nets, nets, losses = 10, [], []
mix_epochs, burnin_epochs = 100, 1000
#num_epochs = mix_epochs*num_nets + burnin_epochs

num_epochs,  nb_train = 2000, len(x_train)

x_extended = np.linspace(-5, 5, int(no_points * 1.5))[:, None]  # More points for smoother visualization
x_extended_tensor = torch.tensor(x_extended, dtype=torch.float32).cpu()

# Train with MALA
MALA_net = MALA_Wrapper(
    input_dim=1,
    output_dim=1,
    no_units=200,
    learn_rate=1e-4,
    weight_decay=0.1,
    init_log_noise=0
)

mala_samples = []
mala_noises = []

acceptance_rates = []

for i in range(num_epochs):
    # Compute the loss and perform a MALA step
    loss, acceptance_rate = MALA_net.fit(x_train_tensor, y_train_tensor)
    acceptance_rates.append(acceptance_rate)

    if i % 200 == 0:
        print(f'Epoch: {i}, Train loss = {loss:.3f}, Acceptance Rate = {acceptance_rate:.3f}')

    if i % mix_epochs == 0 and i > burnin_epochs:
        mala_samples.append(copy.deepcopy(MALA_net.model))

# Collect predictions for MALA (Test Dataset)
mala_test_samples = []
mala_noises_test = []

for network in mala_samples:
    preds = network.forward(x_test_tensor).cpu().data.numpy()
    mala_test_samples.append(preds)
    mala_noises_test.append(torch.exp(network.log_noise).cpu().data.numpy())

mala_test_samples = np.array(mala_test_samples)
mala_noises_test = np.array(mala_noises_test).reshape(-1)

# Compute epistemic and aleatoric uncertainties (Test Dataset)
mala_test_means = mala_test_samples.mean(axis=0).flatten()
mala_test_aleatoric = mala_noises_test.mean()  # Mean predicted noise
mala_test_epistemic = (mala_test_samples.var(axis=0) ** 0.5).flatten()
mala_test_total_unc = (mala_test_aleatoric**2 + mala_test_epistemic**2)**0.5

# Predictions for the Extended Range
mala_extended_samples = []
mala_noises_extended = []

for network in mala_samples:
    preds = network.forward(x_extended_tensor).cpu().data.numpy()
    mala_extended_samples.append(preds)
    mala_noises_extended.append(torch.exp(network.log_noise).cpu().data.numpy())

# Calculate RMSE and NLL for MALA samples (Test Dataset)
rmses = []
log_likelihoods = []
sigma_likelihood = mala_noises_test.mean()  # Mean noise as standard deviation for likelihood

for sample_preds in mala_test_samples:
    # Compute RMSE
    rmse = np.sqrt(mean_squared_error(y_test.flatten(), sample_preds.flatten()))
    rmses.append(rmse)
    
    # Compute Log-Likelihood
    log_likelihood = np.mean(norm.logpdf(y_test.flatten(), loc=sample_preds.flatten(), scale=sigma_likelihood))
    log_likelihoods.append(log_likelihood)

# Compute mean and standard deviation of RMSE and NLL
mean_rmse = np.mean(rmses)
std_rmse = np.std(rmses)
mean_log_likelihood = np.mean(log_likelihoods)
std_log_likelihood = np.std(log_likelihoods)

# Print results
print(f"Mean RMSE: {mean_rmse:.4f} +/- {std_rmse:.4f}")
print(f"Mean Log-Likelihood: {mean_log_likelihood:.4f} +/- {std_log_likelihood:.4f}")

mala_extended_samples = np.array(mala_extended_samples)
mala_noises_extended = np.array(mala_noises_extended).reshape(-1)

# Compute mean and uncertainties for the extended range
mala_means_extended = mala_extended_samples.mean(axis=0).flatten()
mala_aleatoric_extended = mala_noises_extended.mean()  # Mean predicted noise
mala_epistemic_extended = (mala_extended_samples.var(axis=0) ** 0.5).flatten()
mala_total_unc_extended = (mala_aleatoric_extended**2 + mala_epistemic_extended**2)**0.5

# ------------------------------------------------------------
# Plot MALA with Uncertainty
plt.figure(figsize=(6.4, 4.8))
plt.style.use('default')

c = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# Plot training data
plt.scatter(x_train, y_train, s=10, marker='x', color=c[7], alpha=0.5, label='Training data')

# Plot extended uncertainty regions (MALA)
plt.fill_between(
    x_extended.flatten(),
    (mala_means_extended + mala_aleatoric_extended),
    (mala_means_extended + mala_total_unc_extended),
    color=c[0],
    alpha=0.3,
    label='Epistemic + Aleatoric'
)
plt.fill_between(
    x_extended.flatten(),
    (mala_means_extended - mala_total_unc_extended),
    (mala_means_extended - mala_aleatoric_extended),
    color=c[0],
    alpha=0.3
)
plt.fill_between(
    x_extended.flatten(),
    (mala_means_extended - mala_aleatoric_extended),
    (mala_means_extended + mala_aleatoric_extended),
    color=c[1],
    alpha=0.4,
    label='Aleatoric'
)

# Plot mean predictions
plt.plot(x_extended.flatten(), mala_means_extended, color='black', linewidth=2, label='MALA Mean Prediction')

# Configure plot
plt.xlim([-5, 5])
plt.ylim([-5, 7])
plt.xlabel('$x$', fontsize=30)
plt.title('MALA', fontsize=40)
plt.tick_params(labelsize=30)
plt.xticks(np.arange(-5, 6, 2))
plt.yticks(np.arange(-4, 7, 2))
plt.gca().set_yticklabels([])
plt.gca().yaxis.grid(alpha=0.3)
plt.gca().xaxis.grid(alpha=0.3)
plt.legend(fontsize=8, loc='upper right')
plt.tight_layout()
plt.savefig("MALABaselineNN3.png")
plt.show()

# Print MALA acceptance rates
average_acceptance_rate = np.mean(acceptance_rates)
print(f"Average MALA Acceptance Rate: {average_acceptance_rate:.4f}")
