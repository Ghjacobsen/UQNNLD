import pandas as pd
import torch
import matplotlib.pyplot as plt

calories = pd.read_csv('C:/Users/45422/Desktop/Bachelor/DataSets/calories.csv')
exercise = pd.read_csv('C:/Users/45422/Desktop/Bachelor/DataSets/exercise.csv')
df = pd.merge(exercise, calories, on='User_ID')
df = df[df['Calories'] < 300].reset_index()
df['Intercept'] = 1

X = df.loc[:, ['Intercept', 'Duration']]
y = df.loc[:, 'Calories']

X_data = torch.tensor(X.loc[0:15000, 'Duration'].values, dtype=torch.float)
y_data = torch.tensor(y.values[0:15000], dtype=torch.float)

#Standardizing the data
df['Duration'] = (df['Duration'] - df['Duration'].mean()) / df['Duration'].std()
df['Calories'] = (df['Calories'] - df['Calories'].mean()) / df['Calories'].std()

X_standardized = torch.tensor(df.loc[0:15000, 'Duration'].values, dtype=torch.float32)
Y_standardized = torch.tensor(df.loc[0:15000, 'Calories'].values, dtype=torch.float32)

sigma_likelihood_standardized = Y_standardized.std().item()
sigma_likelihood = y_data.std().item()

if __name__ == "__main__":

    # Create a visually appealing scatter plot
    plt.figure(figsize=(6.4, 4.8))
    scatter = plt.scatter(X_data, y_data, c=y_data, cmap='viridis', alpha=1, edgecolor='k', s=10)
    #plt.colorbar(scatter, label='Calories burned (color gradient)')
    plt.xlabel('Exercise Duration (minutes)', fontsize=12)
    plt.ylabel('Calories Burned', fontsize=12)
    plt.title('Scatter Plot of Exercise Duration vs Calories Burned', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig("CalExScatter.png")
    plt.show()


    print(sigma_likelihood)

        # Create a visually appealing scatter plot
    plt.figure(figsize=(6.4, 4.8))
    scatter = plt.scatter(X_standardized, Y_standardized, c=Y_standardized, cmap='viridis', alpha=1, edgecolor='k', s=10)
    #plt.colorbar(scatter, label='Calories burned (color gradient)')
    plt.xlabel('Exercise Duration (minutes)', fontsize=12)
    plt.ylabel('Calories Burned', fontsize=12)
    plt.title('Scatter Plot of Exercise Duration vs Calories Burned', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig("CalExScatter.png")
    plt.show()


    print(sigma_likelihood_standardized)