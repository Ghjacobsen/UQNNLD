import wandb
from Classes import Langevin_Wrapper, SGD_Wrapper, MALA_Wrapper
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import torch
import copy
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from scipy.stats import norm

# Load datasets
exercise = pd.read_csv('C:/Users/45422/Desktop/Bachelor/DataSets/exercise.csv')
calories = pd.read_csv('C:/Users/45422/Desktop/Bachelor/DataSets/calories.csv')

# Merge datasets on the common column (assume "id" is the common column for demonstration)
data = pd.merge(calories, exercise, on="User_ID")

data["Gender"] = data["Gender"].map({"male": 0, "female": 1})

# Select features and labels
features = [
    "Duration", "Heart_Rate", "Body_Temp", "Age", "Height", "Weight", "Gender"
]
label = "Calories"

X = data[features]
y = data[label]

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)



# Function to train and plot for a given wrapper
def train_and_plot(wrapper_class, title, filename, style='langevin'):
    net = wrapper_class(input_dim=X_train.shape[1], output_dim=1, no_units=200, learn_rate=5e-6, init_log_noise=0, weight_decay=0.001)
    num_epochs = 4000
    mix_epochs, burnin_epochs = 100, 3000
    nets = []

    if style in ['sgd', 'langevin']:
        batch_size = len(X_train) // 10

        for epoch in range(num_epochs):
            # Shuffle data for mini-batch training
            perm = np.random.permutation(len(X_train))
            X_train_shuffled = X_train[perm]
            y_train_shuffled = y_train.to_numpy()[perm]

            for i in range(0, len(X_train), batch_size):
                X_batch = X_train_shuffled[i:i + batch_size]
                y_batch = y_train_shuffled[i:i + batch_size]

                # Convert mini-batches to tensors
                X_batch_tensor = torch.tensor(X_batch, dtype=torch.float32)
                y_batch_tensor = torch.tensor(y_batch, dtype=torch.float32).view(-1, 1)

                # Compute the loss
                loss = net.fit(X_batch_tensor, y_batch_tensor)

            # Logging every 200 epochs
            if epoch % 200 == 0:
                print(f'{title} - Epoch: {epoch}, Train loss = {loss:.3f}')

            # Save networks after burn-in
            if epoch % mix_epochs == 0 and epoch > burnin_epochs:
                if hasattr(net, 'network'):
                    nets.append(copy.deepcopy(net.network))
                else:
                    raise AttributeError(f"The {title} wrapper has no 'network' attribute.")
    else:
        for i in range(num_epochs):
            # Call the fit method
            fit_result = net.fit(X_train_tensor, y_train_tensor)

            if isinstance(fit_result, tuple):
                loss, acceptance_rate = fit_result
            else:
                loss = fit_result
                acceptance_rate = None

            if i % 200 == 0:
                if acceptance_rate is not None:
                    print(f'{title} - Epoch: {i}, Train loss = {loss:.3f}, Acceptance Rate = {acceptance_rate:.3f}')
                else:
                    print(f'{title} - Epoch: {i}, Train loss = {loss:.3f}')

            if i % mix_epochs == 0 and i > burnin_epochs:
                if hasattr(net, 'model'):
                    nets.append(copy.deepcopy(net.model))
                elif hasattr(net, 'network'):
                    nets.append(copy.deepcopy(net.network))
                else:
                    raise AttributeError(f"The {title} wrapper has no 'model' or 'network' attribute.")

    # Generate predictions
    samples, noises = [], []
    for network in nets:
        preds = network.forward(X_test_tensor).cpu().data.numpy()
        samples.append(preds)
        noises.append(torch.exp(network.log_noise).cpu().data.numpy())

    samples = np.array(samples)
    noises = np.array(noises).reshape(-1)
    means = samples.mean(axis=0).flatten()
    aleatoric = noises.mean()
    epistemic = (samples.var(axis=0) ** 0.5).flatten()
    total_unc = (aleatoric**2 + epistemic**2)**0.5

    if style in ['langevin', 'mala']:  # Calculate RMSE/NLL with uncertainties
        rmses = []
        nlls = []
        epsilon = 1e-6
        corrected_aleatoric = np.clip(aleatoric, epsilon, None)
        for sample_preds in samples:
            rmse = np.sqrt(mean_squared_error(y_test.to_numpy().flatten(), sample_preds.flatten()))
            rmses.append(rmse)
            log_likelihood = np.mean(norm.logpdf(
                y_test.to_numpy().flatten(),  # True values 
                loc=sample_preds.flatten(),  # Predicted mean
                scale=np.sqrt(corrected_aleatoric**2)  # Standard deviation
            ))
            nlls.append(log_likelihood)

        mean_rmse = np.mean(rmses)
        std_rmse = np.std(rmses)
        mean_log_likelihood = np.mean(nlls)
        std_log_likelihood = np.std(nlls)

        print(f"{title} - Mean RMSE: {mean_rmse:.3f} ± {std_rmse:.3f}")
        print(f"{title} - Mean NLL: {mean_log_likelihood:.3f} ± {std_log_likelihood:.3f}")

    elif style == 'sgd':  # Calculate RMSE/NLL without uncertainties
        sgd_rmse = np.sqrt(mean_squared_error(y_test.to_numpy().flatten(), means.flatten()))
        sgd_log_likelihood = np.mean(norm.logpdf(
            y_test.to_numpy().flatten(), loc=means.flatten(), scale=aleatoric
        ))

        print(f"{title} - RMSE: {sgd_rmse:.3f}")
        print(f"{title} - NLL: {sgd_log_likelihood:.3f}")

    c = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # Random subset of 100 points
    np.random.seed(42)
    subset_size = 40
    indices = np.random.choice(len(means), size=subset_size, replace=False)  # Subset based on predictions

    # Subset predictions, uncertainties, and true values
    subset_means = means[indices]
    subset_total_unc = total_unc[indices]
    subset_y_test = y_test.to_numpy()[indices]  # Convert y_test to NumPy array for proper subsetting

    # Visualization based on the style (Langevin, SGD, MALA)
    if style == 'langevin':
        plt.figure(figsize=(6, 6))
        plt.errorbar(
            subset_y_test,
            subset_means,
            yerr=subset_total_unc,
            fmt='o',
            color=c[0],
            alpha=1,  # Blue circles with no transparency
            label='Predictions',
        )
        plt.errorbar(
            subset_y_test,
            subset_means,
            yerr=subset_total_unc,
            fmt='|',
            color=c[1],
            alpha=0.7,
            label='Uncertainty'
        )
        plt.plot(
            [subset_y_test.min(), subset_y_test.max()],
            [subset_y_test.min(), subset_y_test.max()],
            'r--',
            label='Perfect Prediction',
            alpha = 0.6,
            zorder=10  # Ensure it appears above everything else
        )
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title('SGLD', fontsize=18)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.savefig(filename)
        plt.close()

    elif style == 'sgd':
        plt.figure(figsize=(6, 6))
        plt.plot(
            subset_y_test,
            subset_means,
            'o',
            color=c[0],
            label='Predictions',
            alpha=1
        )
        plt.plot(
            [subset_y_test.min(), subset_y_test.max()],
            [subset_y_test.min(), subset_y_test.max()],
            'r--',
            label='Perfect Prediction',
            alpha = 0.6,
            linewidth=2
        )
        plt.title('SGD', fontsize=18)
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.savefig(filename)
        plt.close()

    elif style == 'mala':
        plt.figure(figsize=(6, 6))
        plt.errorbar(
            subset_y_test,
            subset_means,
            yerr=subset_total_unc,
            fmt='o',
            color=c[0],
            alpha=1,  # Blue circles with no transparency
            label='Predictions',
        )
        plt.errorbar(
            subset_y_test,
            subset_means,
            yerr=subset_total_unc,
            fmt='|',
            color=c[1],
            alpha=0.7,
            label='Uncertainty'
        )
        plt.plot(
            [subset_y_test.min(), subset_y_test.max()],
            [subset_y_test.min(), subset_y_test.max()],
            'r--',
            label='Perfect Prediction',
            alpha = 0.6,
            zorder=10  # Ensure it appears above everything else
        )
        plt.title('MALA', fontsize=18)
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.savefig(filename)
        plt.close()

# Run plots for Langevin (SGLD), SGD, and MALA
#train_and_plot(SGD_Wrapper, "SGD Wrapper", filename='CaloriesSGD', style='sgd')
train_and_plot(Langevin_Wrapper, "Langevin Wrapper (SGLD)", filename='CaloriesSGLD', style='langevin')
#train_and_plot(MALA_Wrapper, "MALA Wrapper", filename='CaloriesMALA', style='mala')