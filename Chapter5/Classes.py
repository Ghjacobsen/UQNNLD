import GPy
import time
import copy
import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Optimizer
from torch.optim.sgd import SGD

from torchvision import datasets, transforms
from torchvision.utils import make_grid
from tqdm import tqdm, trange
#from google.colab import files
#config InlineBackend.figure_format = 'svg'

def to_variable(var=(), cuda=False, volatile=False):
    out = []
    for v in var:
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v).type(torch.FloatTensor)

        if not v.is_cuda and cuda:
            v = v.cpu()  # Replaced .cuda() with .cpu()

        if not isinstance(v, Variable):
            v = Variable(v, volatile=volatile)

        out.append(v)
    return out

class Langevin_SGD(Optimizer):

    def __init__(self, params, lr, weight_decay=0, nesterov=False):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, weight_decay=weight_decay)

        super(Langevin_SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if len(p.shape) == 1 and p.shape[0] == 1:
                    p.data.add_(-group['lr'], d_p)

                else:
                    if weight_decay != 0:
                        d_p.add_(weight_decay, p.data)

                    unit_noise = Variable(p.data.new(p.size()).normal_())

                    p.data.add_(-group['lr'], 0.5*d_p + unit_noise/group['lr']**0.5)

        return loss 

    
def log_gaussian_loss(output, target, sigma, no_dim):
    exponent = -0.5*(target - output)**2/sigma**2
    log_coeff = -no_dim*torch.log(sigma)

    return - (log_coeff + exponent).sum()


def get_kl_divergence(weights, prior, varpost):
    prior_loglik = prior.loglik(weights)

    varpost_loglik = varpost.loglik(weights)
    varpost_lik = varpost_loglik.exp()

    return (varpost_lik*(varpost_loglik - prior_loglik)).sum()


class gaussian:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def loglik(self, weights):
        exponent = -0.5*(weights - self.mu)**2/self.sigma**2
        log_coeff = -0.5*(np.log(2*np.pi) + 2*np.log(self.sigma))

        return (exponent + log_coeff).sum()
    
class Langevin_Layer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Langevin_Layer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.weights = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim).uniform_(-0.01, 0.01))
        self.biases = nn.Parameter(torch.Tensor(self.output_dim).uniform_(-0.01, 0.01))

    def forward(self, x):

        return torch.mm(x, self.weights) + self.biases
    
class Langevin_Model(nn.Module):
    def __init__(self, input_dim, output_dim, no_units, init_log_noise):
        super(Langevin_Model, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.no_units = no_units
        
        # Define the layers of the model
        self.fc1 = nn.Linear(input_dim, no_units)
        self.fc2 = nn.Linear(no_units, no_units)
        self.fc3 = nn.Linear(no_units, output_dim)

        # Activation function
        self.activation = nn.ReLU(inplace=True)
        
        # Initialize log_noise on CPU
        self.log_noise = nn.Parameter(torch.FloatTensor([init_log_noise]))  # Corrected to CPU

    def forward(self, x):
    # Ensure the input is of the shape (batch_size, input_dim)
        x = x.view(-1, self.input_dim)  # Reshape input tensor to match (batch_size, input_dim)
        
        x = self.activation(self.fc1(x))  # Apply the first linear layer and activation
        x = self.activation(self.fc2(x))  # Apply the second linear layer and activation
        return self.fc3(x)  # Output layer

class Langevin_Wrapper:
    def __init__(self, input_dim, output_dim, no_units, learn_rate, init_log_noise, weight_decay):
        self.learn_rate = learn_rate

        self.network = Langevin_Model(input_dim = input_dim, output_dim = output_dim,
                                      no_units = no_units, init_log_noise = init_log_noise)
        self.network.cpu()  # Replaced .cuda() with .cpu()

        self.optimizer = Langevin_SGD(self.network.parameters(), lr=self.learn_rate, weight_decay=weight_decay)
        self.loss_func = log_gaussian_loss

    def fit(self, x, y):
        x, y = to_variable(var=(x, y), cuda=False)  # Ensured variables are on CPU

        # reset gradient and total loss
        self.optimizer.zero_grad()

        output = self.network(x)
        loss = self.loss_func(output, y, torch.exp(self.network.log_noise), 1)

        loss.backward()
        self.optimizer.step()

        return loss
    
class SGD_Wrapper:
    def __init__(self, input_dim, output_dim, no_units, learn_rate, init_log_noise, weight_decay):
        self.learn_rate = learn_rate


        # Define the neural network
        self.network = Langevin_Model(input_dim=input_dim, output_dim=output_dim, no_units=no_units, init_log_noise=0)
        self.network.cpu()  # Ensured network runs on CPU

        # Use standard SGD as optimizer
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=self.learn_rate, weight_decay=weight_decay)
        self.loss_func = log_gaussian_loss

    def fit(self, x, y):
        x, y = to_variable(var=(x, y), cuda=False)  # Ensured variables are on CPU

        # Reset gradient and total loss
        self.optimizer.zero_grad()

        # Forward pass
        output = self.network(x)
        loss = self.loss_func(output, y, torch.exp(self.network.log_noise), 1)

        # Backpropagation and optimizer step
        loss.backward()
        self.optimizer.step()

        return loss

class MALA_Wrapper:
    def __init__(self, input_dim, output_dim, no_units, learn_rate, weight_decay, init_log_noise):
        self.model = Langevin_Model(input_dim, output_dim, no_units, init_log_noise)
        self.learn_rate = learn_rate
        self.weight_decay = weight_decay

        # Define optimizer using MALA_optimizer
        self.optimizer = MALA_optimizer(
            self.model,
            self.model.parameters(),
            learn_rate=self.learn_rate,
            weight_decay=self.weight_decay,
        )

    def fit(self, x_train, y_train):
        self.model.train()

        # Ensure tensors are in the correct type and device
        if not isinstance(x_train, torch.Tensor):
            x_train = torch.tensor(x_train, dtype=torch.float32)
        if not isinstance(y_train, torch.Tensor):
            y_train = torch.tensor(y_train, dtype=torch.float32)

        # Zero gradients
        self.optimizer.zero_grad()

        # Forward pass
        outputs = self.model(x_train)


        noise_variance = torch.exp(self.model.log_noise)  # Convert log_noise to variance

        loss = torch.mean((outputs - y_train) ** 2 / noise_variance + self.model.log_noise)

        #print(f"noisevariance:", noise_variance)
        #print(f"[DEBUG] Loss: {loss.item():.4f}, Data Loss: {data_loss:.4f}, Noise Term: {noise_term:.4f}")

        # Backward pass and parameter update using MALA
        acceptance_rate = self.optimizer.step(loss, x_train, y_train)

        # Debugging: check log_noise gradient and updated value
        #if self.model.log_noise.grad is not None:
            #print(f"[DEBUG] log_noise Gradient Norm: {self.model.log_noise.grad.norm().item():.4f}")
        #else:
            #print("[DEBUG] log_noise Gradient is None!")

        #print(f"[DEBUG] Updated log_noise: {self.model.log_noise.item():.4f}")

        return loss.item(), acceptance_rate



class MALA_optimizer:
    def __init__(self, model, parameters, learn_rate, weight_decay):
        self.model = model
        self.parameters = list(parameters) + [self.model.log_noise]  # Include log_noise
        self.learn_rate = torch.tensor(learn_rate, dtype=torch.float32)
        self.weight_decay = weight_decay

        self.total_proposals = 0
        self.accepted_proposals = 0

    def zero_grad(self):
        for param in self.parameters:
            if param.grad is not None:
                param.grad.zero_()

    def step(self, loss, x_train, y_train):
        # Compute gradients
        loss.backward()

        # Current parameter values
        current_params = [param.clone() for param in self.parameters]

        # Compute proposal
        proposal_params = []
        for param in self.parameters:
            if param.grad is not None:
                grad = param.grad + self.weight_decay * param
                noise = torch.randn_like(param)
                proposal = param - 0.5 * self.learn_rate * grad + torch.sqrt(2 * self.learn_rate) * noise
                proposal_params.append(proposal)
            else:
                proposal_params.append(param.clone())

        # Compute Metropolis-Hastings acceptance step
        accept_prob = self._metropolis_hastings(current_params, proposal_params, x_train, y_train)
        self.total_proposals += 1

        # Accept or reject proposal
        if torch.rand(1).item() < accept_prob:
            self.accepted_proposals += 1
            for param, proposal in zip(self.parameters, proposal_params):
                param.data = proposal.data
        #print(f"Updated log_noise: {self.model.log_noise.item():.4f}")
        #print(f"[DEBUG] Updated log_noise: {self.model.log_noise.item():.4f}")

        acceptance_ratio = self.accepted_proposals / self.total_proposals
        return acceptance_ratio

    def _metropolis_hastings(self, current_params, proposal_params, x_train, y_train):
        # Compute current loss
        self._set_params(current_params)
        outputs_current = self.model(x_train)

        noise_variance = torch.exp(self.model.log_noise)
        current_loss = torch.mean((outputs_current - y_train) ** 2 / noise_variance + self.model.log_noise).item()

        # Compute proposal loss
        self._set_params(proposal_params)
        outputs_proposal = self.model(x_train)
        proposal_loss = torch.mean((outputs_proposal - y_train) ** 2 / noise_variance + self.model.log_noise).item()

        # Compute acceptance probability
        accept_prob = torch.exp(torch.tensor(current_loss - proposal_loss, dtype=torch.float32))
        #print(accept_prob)
        return min(1.0, accept_prob.item())

    def _set_params(self, params):
        # Utility function to set model parameters
        for param, new_value in zip(self.parameters, params):
            param.data = new_value.data

