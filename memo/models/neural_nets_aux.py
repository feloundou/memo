# import numpy as np
# import scipy.signal
# from gym.spaces import Box, Discrete
#
# import numpy as np
# import torch.nn.functional as F
# from torch import nn
# from torch.nn import Parameter
# import torch.nn.functional as F
#
# import torch
#
# from torch.distributions import Independent, OneHotCategorical
# from torch.distributions.normal import Normal
# from torch.distributions.categorical import Categorical
#
# # from utils import *
#
# class GaussianPolicy(nn.Module):
#     def __init__(self, input_dim, hidden_dims, activation, output_activation, action_dim):
#         super(GaussianPolicy, self).__init__()
#         # print("Gaussian policy used.")
#         self.log_std = nn.Parameter(-0.5 * torch.ones(action_dim))
#         self.mu = MLP(layers=[input_dim] + list(hidden_dims) + [action_dim], activation=activation,
#                       output_activation=output_activation)
#
#     def forward(self, x, act=None):
#         policy = Normal(self.mu(x), self.log_std.exp())
#         pi = policy.sample()
#         logp_pi = policy.log_prob(pi).sum(dim=1)
#
#
#         if act is not None:
#             logp = policy.log_prob(act).sum(dim=1)
#
#
#         else:
#             logp = None
#
#         return pi, logp, logp_pi
#
#
#
#
# class Reward_VDB(nn.Module):
#     # def __init__(self, num_inputs, args):
#     def __init__(self, obs_space, act_space, hidden_sizes, activation=nn.Tanh):
#         super().__init__()
#         obs_dim = obs_space.shape[0]
#         act_dim = act_space.shape[0]
#         discrim_dim = obs_dim + act_dim
#
#         # self.mu_net = mlp([discrim_dim] + list(hidden_sizes) + [1], activation)
#
#         z_size = 128
#
#         self.shared_net = mlp([discrim_dim] + list(hidden_sizes), activation)
#         self.mu_net = nn.Linear(hidden_sizes[-1], z_size)
#         self.var_net = nn.Linear(hidden_sizes[-1], z_size)
#
#         # self.fc1 = nn.Linear(num_inputs, args.hidden_size)
#         # self.fc1 = nn.Linear(discrim_dim, hidden_sizes[0])
#         # self.fc2 = nn.Linear(hidden_sizes[0], z_size)
#         # self.fc3 = nn.Linear(hidden_sizes[0], z_size)
#         self.fc4 = nn.Linear(z_size, hidden_sizes[0])
#         self.fc5 = nn.Linear(hidden_sizes[0], 1)
#
#         self.fc5.weight.data.mul_(0.1)
#         self.fc5.bias.data.mul_(0.0)
#
#     def encoder(self, x):
#         h = torch.tanh(self.shared_net(x))
#         return self.mu_net(h), self.var_net(h)
#
#     def reparameterize(self, mu, logvar):
#         std = torch.exp(logvar / 2)
#         eps = torch.randn_like(std)
#         return mu + std * eps
#
#     def discriminator(self, z):
#         h = torch.tanh(self.fc4(z))
#         return torch.sigmoid(self.fc5(h))
#
#     def forward(self, x):
#         mu, logvar = self.encoder(x)
#         z = self.reparameterize(mu, logvar)
#         prob = self.discriminator(z)
#         return prob, mu, logvar
#
#
# class VDB(nn.Module):
#     # def __init__(self, num_inputs, args):
#     def __init__(self, obs_space, act_space, hidden_sizes, activation=nn.Tanh):
#         super().__init__()
#         obs_dim = obs_space.shape[0]
#         act_dim = act_space.shape[0]
#         discrim_dim = obs_dim + act_dim
#         z_size = 128
#
#         # self.fc1 = nn.Linear(num_inputs, args.hidden_size)
#         self.fc1 = nn.Linear(discrim_dim, hidden_sizes[0])
#         self.fc2 = nn.Linear(hidden_sizes[0], z_size)
#         self.fc3 = nn.Linear(hidden_sizes[0], z_size)
#         self.fc4 = nn.Linear(z_size, hidden_sizes[0])
#         self.fc5 = nn.Linear(hidden_sizes[0], 1)
#
#         self.fc5.weight.data.mul_(0.1)
#         self.fc5.bias.data.mul_(0.0)
#
#     def encoder(self, x):
#         h = torch.tanh(self.fc1(x))
#         return self.fc2(h), self.fc3(h)
#
#     def reparameterize(self, mu, logvar):
#         std = torch.exp(logvar / 2)
#         eps = torch.randn_like(std)
#         return mu + std * eps
#
#     def discriminator(self, z):
#         h = torch.tanh(self.fc4(z))
#         return torch.sigmoid(self.fc5(h))
#
#     def forward(self, x):
#         mu, logvar = self.encoder(x)
#         z = self.reparameterize(mu, logvar)
#         prob = self.discriminator(z)
#         return prob, mu, logvar
#
#
# class MLP(nn.Module):
#     def __init__(self, layers, activation=torch.tanh, output_activation=None,
#                  output_squeeze=False):
#         super(MLP, self).__init__()
#         self.layers = nn.ModuleList()
#         self.activation = activation
#         self.output_activation = output_activation
#         self.output_squeeze = output_squeeze
#
#         for i, layer in enumerate(layers[1:]):
#             self.layers.append(nn.Linear(layers[i], layer))
#             nn.init.zeros_(self.layers[i].bias)
#
#     def forward(self, input):
#         x = input
#         for layer in self.layers[:-1]:
#             x = self.activation(layer(x))
#         if self.output_activation is None:
#             x = self.layers[-1](x)
#         else:
#             x = self.output_activation(self.layers[-1](x))
#         return x.squeeze() if self.output_squeeze else x
#
#
# class DiagGaussianLayer(nn.Module):
#     '''
#     Implements a layer that outputs a Gaussian distribution with a diagonal
#     covariance matrix
#     Attributes
#     ----------
#     log_std : torch.FloatTensor
#         the log square root of the diagonal elements of the covariance matrix
#     Methods
#     -------
#     __call__(mean)
#         takes as input a mean vector and outputs a Gaussian distribution with
#         diagonal covariance matrix defined by log_std
#     '''
#
#     def __init__(self, output_dim=None, log_std=None):
#         nn.Module.__init__(self)
#
#         self.log_std = log_std
#
#         if log_std is None:
#             self.log_std = Parameter(torch.zeros(output_dim), requires_grad=True)
#
#     def __call__(self, mean):
#         std = torch.exp(self.log_std)
#         normal_dist = Independent(Normal(loc=mean, scale=std), 1)
#
#         return normal_dist
#
#
# def build_layers(input_dim, hidden_dims, output_dim,
#                  activation=nn.Tanh, output_activation=nn.Identity):
#     '''
#     Returns a list of Linear and Tanh layers with the specified layer sizes
#     Parameters
#     ----------
#     input_dim : int
#         the input dimension of the first linear layer
#     hidden_dims : list
#         a list of type int specifying the sizes of the hidden layers
#     output_dim : int
#         the output dimension of the final layer in the list
#     Returns
#     -------
#     layers : list
#         a list of Linear layers, each one followed by a Tanh layer, excluding the
#         final layer
#     '''
#
#     layer_sizes = [input_dim] + hidden_dims + [output_dim]
#     layers = []
#
#     for i in range(len(layer_sizes) - 1):
#         act = activation if i < len(layer_sizes) - 2 else output_activation # Tyna note
#         layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1], bias=True))
#
#         if i != len(layer_sizes) - 2:
#             layers.append(nn.Tanh())
#
#     return layers
#
#
#
# def MLP_DiagGaussianPolicy(state_dim, hidden_dims, action_dim,
#                            log_std=None):
#     '''
#     Build a multilayer perceptron with a DiagGaussianLayer at the output layer
#     Parameters
#     ----------
#     state_dim : int
#         the input size of the network
#     hidden_dims : list
#         a list of type int specifying the sizes of the hidden layers
#     action_dim : int
#         the dimensionality of the Gaussian distribution to be outputted by the
#         policy
#     log_std : torch.FloatTensor
#         the log square root of the diagonal elements of the covariance matrix
#         (will be set to a vector of zeros if none is specified)
#     Returns
#     -------
#     policy : torch.nn.Sequential
#         a pytorch sequential model that outputs a Gaussian distribution
#     '''
#
#     layers = build_layers(state_dim, hidden_dims, action_dim)
#     layers[-1].weight.data *= 0.1
#     layers[-1].bias.data *= 0.0
#     layers.append(DiagGaussianLayer(action_dim, log_std))
#     policy = nn.Sequential(*layers)
#
#     return policy
#
#
# class ActorCritic(nn.Module):
#     def __init__(self, input_dim, action_space, hidden_dims=(64, 64), activation=torch.tanh, output_activation=None,
#                  policy=None):
#         super(ActorCritic, self).__init__()
#
#         if policy is None:
#             if isinstance(action_space, Box):
#                 self.pi = GaussianPolicy(input_dim, hidden_dims, activation, output_activation,
#                                              action_space.shape[0])
#             elif isinstance(action_space, Discrete):
#                 self.pi = CategoricalPolicy(input_dim, hidden_dims, activation, output_activation, action_space.n)
#         else:
#             self.pi = policy(input_dim, hidden_dims, activation, output_activation, action_space)
#
#         self.v = MLP(layers=[input_dim] + list(hidden_dims) + [1], activation=activation, output_squeeze=True)
#
#     def forward(self, x, a=None):
#         pi, logp, logp_pi = self.pi(x, a)
#         v = self.v(x)
#
#         return pi, logp, logp_pi, v
