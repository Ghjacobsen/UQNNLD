
# Uncertainty Quantification for Neural Networks Using Langevin Dynamics

This repository contains the codebase for the **bachelor's thesis** titled *"Uncertainty Quantification for Neural Networks Using Langevin Dynamics"*, written at **DTU Compute**. The project explores methods for incorporating uncertainty quantification in machine learning, with a focus on applying Langevin dynamics for Bayesian inference in neural networks.

## Repository Structure

The repository is organized into three main folders, each corresponding to a specific part of the thesis:

### 1. **Chapter2&3: Preliminaries**
   - This folder contains the code related to the foundational concepts introduced in Chapters 2 and 3 of the thesis.
   - Topics covered include:
     - Optimization methods (e.g., SGD).
     - Introduction to sampling methods such as Langevin dynamics.
   - These scripts provide the groundwork for understanding the more advanced topics in subsequent chapters.

### 2. **Chapter4: Bayesian Linear Regression**
   - This folder contains the code implementing Bayesian Linear Regression (BLR).
   - Key features include:
     - Demonstrations of how Bayesian inference works in simpler linear models.
     - Comparison of performance metrics like RMSE and NLL for BLR.

### 3. **Chapter5: Neural Networks**
   - This folder contains the code for applying Langevin dynamics to neural networks.
   - Topics include:
     - Training and uncertainty quantification in neural networks using methods like SGLD and MALA.
     - Evaluation of the algorithms' performance on benchmark datasets.
   - This section highlights the practical application of Langevin dynamics in deep learning.

## Requirements

To run the code in this repository, the following packages are required:

- Python 3.8 or later
- NumPy
- Matplotlib
- PyTorch
- Scikit-learn
- Weights & Biases (optional, for experiment tracking)

Install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

## How to Use

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/uncertainty-quantification.git
   cd uncertainty-quantification
   ```

2. Navigate to the desired chapter folder and execute the relevant scripts:

   ```bash
   cd Chapter4
   python bayesian_linear_regression.py
   ```

3. Explore the results and visualizations generated in each section.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project was developed as part of a bachelor's thesis at DTU Compute. Special thanks to the advisors and peers who supported the work.

## Contact

For questions or suggestions, feel free to contact:

- **Author Name**: [Your Email]
- GitHub: [Your GitHub Profile]

Enjoy exploring uncertainty quantification in machine learning! 🎉
