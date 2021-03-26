# import scipy.signal
from gym.spaces import Box, Discrete
#
import numpy as np
import torch
from torch import nn
# from torch.nn import Parameter
import torch.nn.functional as F
#
#
from torch.distributions import Independent, OneHotCategorical, Categorical
from torch.distributions.normal import Normal
# # from torch.distributions.categorical import Categorical
#
# # from utils import *
#
#
#
def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and optionally
        # compute the log likelihood of given actions under those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()

        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        print("Actor Crit std", std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)  # Last axis sum needed for Torch Normal distribution


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1)  # Critical to ensure v has right shape.


class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space,
                 hidden_sizes=(64, 64), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]
        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # build value function critics
        self.v = MLPCritic(obs_dim, hidden_sizes, activation)
        self.vc = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            print("pi dist! ", pi)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
            vc = self.vc(obs)

        return a.numpy(), v.numpy(), vc.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]


#
# class ValorActorCritic(nn.Module):
#
#     def __init__(self, input_dim, action_space,
#                  hidden_sizes=(64, 64), activation=nn.Tanh):
#         super().__init__()
#
#         obs_dim = input_dim
#         # policy builder depends on action space
#         if isinstance(action_space, Box):
#             self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
#         elif isinstance(action_space, Discrete):
#             self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)
#
#         # build value function critics
#         self.v = MLPCritic(obs_dim, hidden_sizes, activation)
#         self.vc = MLPCritic(obs_dim, hidden_sizes, activation)
#
#     def step(self, obs):
#         with torch.no_grad():
#             print("PI dist: ", self.pi)
#             pi = self.pi._distribution(obs)
#             a = pi.sample()
#             logp_a = self.pi._log_prob_from_distribution(pi, a)
#             v = self.v(obs)
#             vc = self.vc(obs)
#
#         return a.numpy(), v.numpy(), vc.numpy(), logp_a.numpy()
#
#     def act(self, obs):
#         return self.step(obs)[0]
#
#
#
#
# class GaussianReward(nn.Module):
#     def __init__(self, obs_dim, hidden_sizes, activation):
#         super().__init__()
#
#         self.shared_net = mlp([obs_dim] + list(hidden_sizes), activation)
#         self.mu_net = nn.Linear(hidden_sizes[-1], 1)
#         self.var_net = nn.Linear(hidden_sizes[-1], 1)
#
#     def forward(self, x):
#
#         out = F.leaky_relu(self.shared_net(x))
#         mu = self.mu_net(out)
#         std = self.var_net(out)
#         return Normal(loc=mu, scale=std).rsample()
#
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
#
# class GaussianActor(nn.Module):
#     def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
#         super().__init__()
#         # log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
#         # self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
#         # self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)
#         self.shared_net = mlp([obs_dim] + list(hidden_sizes), activation)
#         self.mu_net = nn.Linear(hidden_sizes[-1], act_dim)
#         self.var_net = nn.Linear(hidden_sizes[-1], act_dim)
#
#     def forward(self, x):
#         mu = self.mu_net(F.leaky_relu(self.shared_net(x)))
#         std = self.var_net(F.leaky_relu(self.shared_net(x)))
#         return Normal(loc=mu, scale=std).rsample()
#
#
#
# class DistilledGaussianActor(nn.Module):
#     def __init__(self, obs_dim, act_dim, hidden_sizes, activation, n_experts):
#         super().__init__()
#         obs_dim_aug = obs_dim + n_experts
#         self.shared_net = mlp([obs_dim_aug] + list(hidden_sizes), activation)
#
#         self.mu_net = nn.Linear(hidden_sizes[-1], act_dim)
#         self.var_net = nn.Linear(hidden_sizes[-1], act_dim)
#
#     def forward(self, x):
#
#         out = F.leaky_relu(self.shared_net(x))
#         mu = self.mu_net(out)
#         std = self.var_net(out)
#
#         return Normal(loc=mu, scale=std).rsample()
#
#
#
# class BiclassificationPolicy(nn.Module):
#     def __init__(self, input_dim, hidden_dims, activation, output_activation):
#         super(BiclassificationPolicy, self).__init__()
#
#         self.output_dim = 2
#         self.logits = MLP(layers=[input_dim] + list(hidden_dims) + [self.output_dim], activation=activation)
#
#     def forward(self, x, label=None):
#         logits = self.logits(x)
#         policy = Categorical(logits=logits)
#         l = policy.sample()
#         logp_l = policy.log_prob(l).squeeze()
#         if label is not None:
#             logp = policy.log_prob(label).squeeze()
#         else:
#             logp = None
#
#         return l, logp, logp_l
#
# class Discriminator(nn.Module):
#     def __init__(self, input_dim, hidden_dims=(64, 64), activation=torch.relu, output_activation=torch.softmax):
#         super(Discriminator, self).__init__()
#
#         self.pi = BiclassificationPolicy(input_dim, hidden_dims, activation, output_activation)
#
#     def forward(self, state, gt=None):
#         label, loggt, logp = self.pi(state, gt)
#         return label, loggt, logp
#
#
# class MLPDiscriminator(nn.Module):
#     def __init__(self, obs_space, act_space, hidden_sizes, activation=nn.Tanh):
#         super().__init__()
#         obs_dim = obs_space.shape[0]
#         act_dim = act_space.shape[0]
#         discrim_dim = obs_dim + act_dim
#         self.discrim_net = mlp([discrim_dim] + list(hidden_sizes) + [1], activation)
#
#
#     def forward(self, obs):
#         prob = torch.sigmoid(self.discrim_net(obs))
#         return prob
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
#
# class ValorFFNNPolicy(nn.Module):
#     def __init__(self, input_dim, hidden_dims, activation, output_activation, con_dim):
#         super(ValorFFNNPolicy, self).__init__()
#
#         self.context_net = mlp([input_dim] + list(hidden_dims), activation)
#         self.linear = nn.Linear(hidden_dims[-1], con_dim)
#
#     def forward(self, seq, gt=None, classes=False):
#         logit_seq = self.context_net(seq)
#         self.logits = torch.mean(logit_seq, dim=1)
#         policy = Categorical(logits=self.logits)
#         label = policy.sample()
#         # print("LABEL: ", label)
#         logp = policy.log_prob(label).squeeze()
#
#         if gt is not None:
#             loggt = policy.log_prob(gt).squeeze()
#         else:
#             loggt = None
#
#         if classes is False:
#             return label, loggt, logp
#         else:
#             return label, loggt, logp, gt
#
#
# class ValorDiscriminator(nn.Module):
#     def __init__(self, input_dim, context_dim, activation=nn.Softmax,
#                  output_activation=nn.Softmax, hidden_dims=64):
#
#         super(ValorDiscriminator, self).__init__()
#         self.context_dim = context_dim
#
#         self.pi = ValorFFNNPolicy(input_dim, hidden_dims, activation=nn.Tanh,
#                                   output_activation=nn.Tanh, con_dim=self.context_dim)
#
#     def forward(self, seq, gt=None, classes=False):
#         if classes is False:
#             pred, loggt, logp = self.pi(seq, gt, classes)
#             return pred, loggt, logp
#
#         else:
#             pred, loggt, logp, gt = self.pi(seq, gt, classes)
#             return pred, loggt, logp, gt
#
#
#
#
# class ModValorFFNNPolicy(nn.Module):
#     def __init__(self, input_dim, hidden_dims, activation, output_activation, con_dim):
#         super(ModValorFFNNPolicy, self).__init__()
#
#         self.context_net = mlp([input_dim] + list(hidden_dims) + [con_dim], activation)
#
#
#     def forward(self, seq, gt=None):
#
#         logit_seq = self.context_net(seq)
#         self.logits = torch.mean(logit_seq, dim=1)
#         policy = Categorical(logits=self.logits)
#         label = policy.sample()
#         print("LABEL: ", label)
#         logp = policy.log_prob(label).squeeze()
#
#         if gt is not None:
#             loggt = policy.log_prob(gt).squeeze()
#         else:
#             loggt = None
#
#             return label, loggt, logp
#
#
#
#
# class ModValorDiscriminator(nn.Module):
#     def __init__(self, input_dim, context_dim, activation=nn.Softmax,
#                  output_activation=nn.Softmax, hidden_dims=64):
#
#         super(ModValorDiscriminator, self).__init__()
#         self.context_dim = context_dim
#
#         self.pi = ModValorFFNNPolicy(input_dim, hidden_dims, activation=nn.Tanh,
#                                   output_activation=nn.Tanh, con_dim=self.context_dim)
#
#     def forward(self, seq, gt=None):
#             pred, loggt, logp = self.pi(seq, gt)
#             return pred, loggt, logp
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
# #############################################################################
#
#
#
# class OneHotCategoricalActor(Actor):
#     def __init__(self, obs_dim, con_dim, hidden_sizes, activation):
#         super().__init__()
#         self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [con_dim], activation)
#
#     def _distribution(self, obs):
#         logits = self.logits_net(obs)
#         # print("Categorical logits: ", logits)
#         return OneHotCategorical(logits=logits)
#         # return Categorical(logits=logits)
#
#
#
#
# #####
# class Old_OneHotCategoricalActor(Actor):
#
#     def __init__(self, obs_dim, con_dim, hidden_sizes, activation):
#         super().__init__()
#         self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [con_dim], activation)
#
#     def _distribution(self, obs):
#         logits = self.logits_net(obs)
#         return OneHotCategorical(logits=logits)
#
#     def _log_prob_from_distribution(self, pi, act):
#         return pi.log_prob(act)
#
# class Old_VAE_Encoder(nn.Module):
#     def __init__(self, nx_sizes):
#         super(Old_VAE_Encoder, self).__init__()
#         self.logits_net = mlp(nx_sizes, activation=nn.Tanh)
#
#     def forward(self, x):
#         z = self.logits_net(x)
#         return z
#
#
# class Old_VAE_Decoder(nn.Module):
#     def __init__(self, in_dim, hidden, out_dim, activation=nn.Tanh):
#         super(Old_VAE_Decoder, self).__init__()
#         self.pi = MLPGaussianActor(in_dim, out_dim, hidden, activation)
#
#     def forward(self, x):
#         with torch.no_grad():              # no backprop necessary. Also really useful for getting the right distribution.
#             self.dist = self.pi._distribution(x)
#         return self.dist
#
# class Old_Lambda(nn.Module):
#     """Lambda module converts output of encoder to latent vector
#     :param hidden_size: hidden size of the encoder
#     :param latent_length: latent vector length
#     """
#     def __init__(self, input_dim, latent_length):
#         super(Old_Lambda, self).__init__()
#         self._latent_net = nn.Linear(input_dim, latent_length)   ## old
#
#         hidden_sizes = [500]
#         con_dim = 2
#
#         self.lambda_pi = OneHotCategoricalActor(input_dim, con_dim, hidden_sizes, activation=nn.Tanh)
#
#
#     def forward(self, cell_output):
#         return self.lambda_pi._distribution(cell_output)
#
#
# class VAELOR(nn.Module):
#     """Reimplement old valor model with new fixes"""
#     def __init__(self, obs_dim, latent_dim):
#         super(VAELOR, self).__init__()
#         """
#         Given input tensor, forward prop to get context samples, raw state and state differences.
#         Returns latent variable
#         :param obs_dim: state dimension
#         :param latent_length: latent vector length
#         """
#         act_dim = 2
#         encoder_sizes = [obs_dim, 100, 400]
#         decoder_hidden = [128] * 4
#
#         self.encoder = Old_VAE_Encoder(encoder_sizes)  # original
#         self.lmbd = Old_Lambda(input_dim=encoder_sizes[-1], latent_length=latent_dim)
#         self.decoder = Old_VAE_Decoder(obs_dim + latent_dim, decoder_hidden, act_dim, activation=nn.Tanh)
#
#     def forward(self, state, delta_state, action, latent_labels=None):
#         delta_state_enc = self.encoder(delta_state)  # original
#         latent_v_dist = self.lmbd(delta_state_enc)
#         if latent_labels is None:
#             latent_labels = latent_v_dist.sample()
#
#         # print("Latent V Sample: ", latent_labels[:2])
#         concat_state = torch.cat([state, latent_labels], dim=1)
#         action_dist = self.decoder(concat_state)
#         return action_dist, latent_labels, latent_v_dist
#
#     def compute_latent_loss(self, X, Delta_X, A, context_sample):
#         """
#         Given input tensor, forward propagate, compute the loss, and backward propagate.
#         Represents the lifecycle of a single iteration
#         :param X: Input tensor
#         :return: total loss, reconstruction loss, kl-divergence loss and original input
#         """
#         # decoded_action, latent_labels, logp_action = self(X, Delta_X)
#         action_dist, latent_labels, latent_labels_dist = self(X, Delta_X, A)
#
#         # get latent labels for checking accuracy
#         context_loss = 0.5 - latent_labels_dist.log_prob(context_sample)  # this is the correct version  ##
#         recon_loss = torch.exp(action_dist.log_prob(A).sum(axis=-1))
#         loss = recon_loss * context_loss  # loss = recon_loss + context_loss
#
#         return loss, recon_loss, context_loss, X, latent_labels
#
#
# class Lambda(nn.Module):
#     """Lambda module converts output of encoder to latent vector
#     :param hidden_size: hidden size of the encoder
#     :param latent_length: latent vector length
#     """
#     def __init__(self, hidden_dim, hidden_sizes, con_dim):
#         super(Lambda, self).__init__()
#
#         self.latent_length = con_dim
#         self.lambda_pi = OneHotCategoricalActor(hidden_dim, self.latent_length, hidden_sizes, activation=nn.Tanh)  # original (before reparam)
#         # self.lambda_pi = OneHotCategoricalActor(interm_, self.latent_length, hidden_sizes, activation=nn.Tanh)
#
#     def forward(self, encoder_output):
#         return self.lambda_pi._distribution(encoder_output)
#         # return self.lambda_pi._distribution(reparam_output), reparam_latent_loss
#
#
# class VAE_Encoder(nn.Module):
#     def __init__(self, network_sizes):
#         super(VAE_Encoder, self).__init__()
#
#         self.logits_net = mlp(network_sizes, activation=nn.Tanh)
#
#     def forward(self, x):
#         return self.logits_net(x)
#
# # Try NN.Identity
# class VAE_Decoder(nn.Module):
#     def __init__(self, in_dim, hidden, out_dim, activation):
#         super(VAE_Decoder, self).__init__()
#         self.pi = MLPGaussianActor(in_dim, out_dim, hidden, activation)
#
#     def forward(self, x):
#         with torch.no_grad():              # no backprop necessary. Also really useful for getting the right distribution.
#             self.dist = self.pi._distribution(x)
#             print("decoder action dist", self.dist)
#         return self.dist
#
#
#
class MEMOActor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(MEMOActor, self).__init__()
        # activation = nn.Tanh()
        activation = nn.Tanh()

        self.mu_net = nn.Sequential(
                        nn.Linear(state_dim, 512),
                        nn.ReLU(),
                        # nn.Tanh(),
                        nn.Linear(512, 512),
                        nn.ReLU(),
                        # nn.Tanh(),
                        nn.Linear(512, 512),
                        nn.ReLU(),
                        nn.Tanh(),
                        nn.Linear(512, action_dim))
                        # nn.LeakyReLU())
                        # nn.Tanh())

        log_std = -0.5 * np.ones(action_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))


    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def forward(self, obs):
        # with torch.no_grad():
        mu = self.mu_net(obs)
        # print("Mu: ", mu)
        std = torch.exp(self.log_std)
        # print("Std: ", std)
        return Normal(mu, std)



class MEMO(torch.nn.Module):
    """Multiple Experts, Multiple Objectives;
    """
    def __init__(self, obs_dim, latent_dim, out_dim, encoder_hidden, decoder_hidden):
        super(MEMO, self).__init__()
        """
        Given input tensor, forward prop to get context samples, raw state and state differences.
        Returns latent variable
        :param obs_dim: state dimension
        :param latent_length: latent vector length
        """
        encoder_sizes = [obs_dim] + encoder_hidden

        self.num_embeddings = 10 #10  # latent dimension   # num_embeddings = 2  # 10  # latent dimension
        # self.num_embeddings = 2
        self.embedding_dim = obs_dim
        self.vq_encoder = VQEncoder(obs_dim, encoder_sizes[-1])  # original

        self.prenet = nn.Linear(encoder_sizes[-1], self.embedding_dim)
        self.vector_quantizer = VectorQuantizer(self.num_embeddings, self.embedding_dim)
        self.postnet = nn.Linear(self.embedding_dim, encoder_sizes[-1])
        # self.vq_decoder = VQDecoder(encoder_sizes[-1], 200)
        self.vq_decoder = VQDecoder(encoder_sizes[-1], obs_dim)
        ###

        # self.encoder = VAE_Encoder(encoder_sizes) # original
        # self.decoder = VAE_Decoder(obs_dim + 1, decoder_hidden, out_dim, activation=nn.Tanh)
        self.decoder = MEMOActor(state_dim=obs_dim+1, action_dim=out_dim)
        self.action_vq_dist = None


    def compute_quantized_loss(self, state, delta_state, actions, con_dim):
        delta_state_enc = self.vq_encoder(delta_state)
        encoder_output = self.prenet(delta_state_enc)
        quantized, categorical_proposal = self.vector_quantizer(encoder_output)

        # Straight Through Estimator (Some Magic)
        st_quantized = encoder_output + (quantized - encoder_output).detach()
        post_quantized = self.postnet(st_quantized)
        # print("Post Quantized: ", post_quantized)

        reconstruction = self.vq_decoder(post_quantized)
        # print("Reconstruction: ", reconstruction)

        categorical_proposal_reshape = torch.reshape(categorical_proposal, (-1, 1))
        concat_state_vq = torch.cat([state, categorical_proposal_reshape], dim=-1)
        action_vq_dist = self.decoder(concat_state_vq)
        # print("Action Distribution: ", action_vq_dist)
        ####

        return encoder_output, quantized, reconstruction, categorical_proposal, action_vq_dist
        # removing latent_vq_dist for lambda

    def act(self, state, context_label):
        concat_state_vq = torch.cat([state, torch.reshape(torch.as_tensor(context_label), (-1,))], dim=-1)
        action_vq_dist = self.decoder(concat_state_vq)
        action = action_vq_dist.sample()
        return action

    def forward(self, X, Delta_X, A, context_sample, con_dim, kl_beta=1., recon_gamma=1.):
        """
        Given input tensor, forward propagate, compute the loss, and backward propagate.
        Represents the lifecycle of a single iteration
        :param x: Raw state tensor
        :param delta_x: State difference tensor
        :param a: Action tensor
        :param context_sample: randomly generated one-hot encoded context label proposals
        :param kl_beta: KL divergence temperance factor
        : Important to note that both recon and context loss cannot be negative.
        """
        encoder_output, quantized, reconstruction, vq_latent_labels, action_vq_dist =\
            self.compute_quantized_loss(X, Delta_X, A, con_dim)

        vq_criterion = VQCriterion(beta=kl_beta)
        vq_total_loss, recons_loss, vq_loss, commitment_loss = vq_criterion(Delta_X, encoder_output, quantized, reconstruction)

        # original formula
        # recon_loss = torch.exp(action_vq_dist.log_prob(A).sum(axis=-1))  # kl_beta*kl_beta
        recon_loss = -action_vq_dist.log_prob(A).sum(axis=-1)# kl_beta*kl_beta  ### THIS RECON LOSS WORKS
        # recon_loss = -torch.exp(action_vq_dist.log_prob(A).sum(axis=-1))
        recon_loss *= recon_gamma
        # kl_beta = 1-recon_gamma
        # kl_beta = 1

        vq_cat_loss = vq_total_loss*kl_beta
        loss = recon_loss * vq_cat_loss

        return loss, recon_loss, X, vq_latent_labels, vq_total_loss




class VQEncoder(nn.Module):

    def __init__(self, in_dim, out_dim,):
        super(VQEncoder, self).__init__()
        # self.logits_net = mlp([in_dim, in_dim//2, out_dim], activation=nn.Tanh)
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim // 2),
            # nn.ReLU(inplace=True),
            nn.Tanh(),  # much better than relu
            nn.Linear(out_dim // 2, out_dim)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.net(input)


class Clamper(nn.Module):
    def __init__(self, min=None, max=None):
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, input):
        return torch.clamp(input, self.min, self.max)

class VQDecoder(nn.Module):
    def __init__(self, obs_dim, hidden_size):
        super().__init__()

        test_size = 100
        # test_size=40

        self.net = nn.Sequential(
            nn.Linear(obs_dim, test_size),
            nn.Tanh(),
            # nn.ReLU(inplace=True),
            # nn.Linear(hidden_size, hidden_size),
            # nn.Linear(test_size, test_size),
            # nn.Tanh(),
            # nn.ReLU(inplace=True),
            nn.Linear(test_size, hidden_size),
            nn.Tanh(),
            # nn.LeakyReLU(inplace=True)
            # Clamper(-10, 10),
            # Clamper(-1, 1),
            # nn.Sigmoid()
        )
    def forward(self, input):
        return self.net(input)


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)

        self.scale = 1. / self.num_embeddings
        print("Quantizer Scale: ", self.scale)
        torch.nn.init.uniform_(self.embeddings.weight, -self.scale, self.scale)

    def proposal_distribution(self, input):
        input_shape = input.shape

        flatten_input = input.flatten(end_dim=-2).contiguous()

        distances = (flatten_input ** 2).sum(dim=1, keepdim=True)
        distances = distances + (self.embeddings.weight ** 2).sum(dim=1)
        distances -= 2 * flatten_input @ self.embeddings.weight.t()

        categorical_posterior = torch.argmin(distances, dim=-1, keepdim=True).view(input_shape[:-1])
        # print("Categorical posterior: ", categorical_posterior)

        return categorical_posterior

    def forward(self, input):
        proposal = self.proposal_distribution(input)
        quantized = self.embeddings(proposal).contiguous()

        return quantized, proposal


class VQCriterion(nn.Module):
    """
    vq_loss: \| \text{sg}[I(x, e)] * e  - \text{sg}[z_e(x)] \|_2^2
    """

    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def forward(self, input, encoder_output, quantized, reconstruction):
        flatten_quantized = quantized.flatten(end_dim=-2)
        flatten_encoder_output = encoder_output.flatten(end_dim=-2)

        reconstruction_loss = F.mse_loss(input, reconstruction)

        vq_loss = F.mse_loss(flatten_encoder_output.detach(), flatten_quantized)
        commitment_loss = F.mse_loss(flatten_encoder_output, flatten_quantized.detach())

        total_loss = reconstruction_loss + vq_loss + self.beta * commitment_loss   # Original. TODO: review this loss.
        # total_loss = reconstruction_loss + self.beta * commitment_loss

        return total_loss, reconstruction_loss, vq_loss, commitment_loss

#
# # class VQVAELOR(nn.Module):
# #     """
# #     https://arxiv.org/abs/1711.00937
# #     """
# #     def __init__(self, obs_dim, hidden_dim, out_dim,  num_embeddings, embedding_dim):
# #         super().__init__()
# #
# #         in_dim = obs_dim
# #         hidden_dim = 20
# #
# #         latent_dim = 2
# #
# #         self.encoder = VQEncoder(in_dim, hidden_dim)
# #         self.decoder = VQDecoder(hidden_dim, out_dim)
# #         self.vector_quantizer = VectorQuantizer(num_embeddings, embedding_dim)
# #
# #         self.prenet = nn.Linear(hidden_dim, latent_dim)
# #         self.postnet = nn.Linear(latent_dim, hidden_dim)
# #
# #         # self.prenet = nn.Conv2d(hidden_channels, embedding_dim, kernel_size=1)
# #         # self.postnet = nn.Conv2d(embedding_dim, hidden_channels, kernel_size=3, padding=1)
# #
# #     def encode(self, input: torch.Tensor) -> torch.Tensor:
# #         encoder_output = self.encoder(input)
# #         encoder_output = self.prenet(encoder_output)
# #         quantized = self.vector_quantizer(encoder_output)
# #
# #         return quantized
# #
# #     def forward(self, input: torch.Tensor) -> torch.Tensor:
# #         encoder_output = self.encoder(input)
# #         encoder_output = self.prenet(encoder_output)
# #
# #         quantized = self.vector_quantizer(encoder_output)
# #
# #         # Straight Through Estimator (Some Magic)
# #         st_quantized = encoder_output + (quantized - encoder_output).detach()
# #         post_quantized = self.postnet(st_quantized)
# #
# #         reconstruction = self.decoder(post_quantized)
# #
# #         return encoder_output, quantized, reconstruction
# #
# #     @torch.no_grad()
# #     def generate(self, prior):
# #         raise NotImplementedError()
#
#
#
#
#
# ########
#
# # class VectorQuantizedVAE(nn.Module):
# #     def __init__(self, input_dim, dim, K=512):
# #         super().__init__()
# #         self.encoder = nn.Sequential(
# #             nn.Conv2d(input_dim, dim, 4, 2, 1),
# #             nn.BatchNorm2d(dim),
# #             nn.ReLU(True),
# #             nn.Conv2d(dim, dim, 4, 2, 1),
# #             ResBlock(dim),
# #             ResBlock(dim),
# #         )
# #
# #         self.codebook = VQEmbedding(K, dim)
# #
# #         self.decoder = nn.Sequential(
# #             ResBlock(dim),
# #             ResBlock(dim),
# #             nn.ReLU(True),
# #             nn.ConvTranspose2d(dim, dim, 4, 2, 1),
# #             nn.BatchNorm2d(dim),
# #             nn.ReLU(True),
# #             nn.ConvTranspose2d(dim, input_dim, 4, 2, 1),
# #             nn.Tanh()
# #         )
# #
# #         self.apply(weights_init)
# #
# #     def encode(self, x):
# #         z_e_x = self.encoder(x)
# #         latents = self.codebook(z_e_x)
# #         return latents
# #
# #     def decode(self, latents):
# #         z_q_x = self.codebook.embedding(latents).permute(0, 3, 1, 2)  # (B, D, H, W)
# #         x_tilde = self.decoder(z_q_x)
# #         return x_tilde
# #
# #     def forward(self, x):
# #         z_e_x = self.encoder(x)
# #         z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
# #         x_tilde = self.decoder(z_q_x_st)
# #         return x_tilde, z_e_x, z_q_x
# #
# #
# # class GatedActivation(nn.Module):
# #     def __init__(self):
# #         super().__init__()
# #
# #     def forward(self, x):
# #         x, y = x.chunk(2, dim=1)
# #         return F.tanh(x) * F.sigmoid(y)
# #
# #
# # class GatedMaskedConv2d(nn.Module):
# #     def __init__(self, mask_type, dim, kernel, residual=True, n_classes=10):
# #         super().__init__()
# #         assert kernel % 2 == 1, print("Kernel size must be odd")
# #         self.mask_type = mask_type
# #         self.residual = residual
# #
# #         self.class_cond_embedding = nn.Embedding(
# #             n_classes, 2 * dim
# #         )
# #
# #         kernel_shp = (kernel // 2 + 1, kernel)  # (ceil(n/2), n)
# #         padding_shp = (kernel // 2, kernel // 2)
# #         self.vert_stack = nn.Conv2d(
# #             dim, dim * 2,
# #             kernel_shp, 1, padding_shp
# #         )
# #
# #         self.vert_to_horiz = nn.Conv2d(2 * dim, 2 * dim, 1)
# #
# #         kernel_shp = (1, kernel // 2 + 1)
# #         padding_shp = (0, kernel // 2)
# #         self.horiz_stack = nn.Conv2d(
# #             dim, dim * 2,
# #             kernel_shp, 1, padding_shp
# #         )
# #
# #         self.horiz_resid = nn.Conv2d(dim, dim, 1)
# #
# #         self.gate = GatedActivation()
# #
# #     def make_causal(self):
# #         self.vert_stack.weight.data[:, :, -1].zero_()  # Mask final row
# #         self.horiz_stack.weight.data[:, :, :, -1].zero_()  # Mask final column
# #
# #     def forward(self, x_v, x_h, h):
# #         if self.mask_type == 'A':
# #             self.make_causal()
# #
# #         h = self.class_cond_embedding(h)
# #         h_vert = self.vert_stack(x_v)
# #         h_vert = h_vert[:, :, :x_v.size(-1), :]
# #         out_v = self.gate(h_vert + h[:, :, None, None])
# #
# #         h_horiz = self.horiz_stack(x_h)
# #         h_horiz = h_horiz[:, :, :, :x_h.size(-2)]
# #         v2h = self.vert_to_horiz(h_vert)
# #
# #         out = self.gate(v2h + h_horiz + h[:, :, None, None])
# #         if self.residual:
# #             out_h = self.horiz_resid(out) + x_h
# #         else:
# #             out_h = self.horiz_resid(out)
# #
# #         return out_v, out_h
#
#
#
#
#
# class MLPContextLabeler(Actor):
#
#     # def __init__(self, obs_dim, context_dim, hidden_sizes, activation):
#     def __init__(self, input_dim, context_dim, hidden_sizes, activation):
#         super().__init__()
#         self.logits_net = mlp([input_dim] + list(hidden_sizes) + [context_dim], activation)
#
#     def _distribution(self, obs):
#         logits = self.logits_net(obs)
#         return Categorical(logits=logits)
#
#
#     def _log_prob_from_distribution(self, pi, con):
#         return pi.log_prob(con)
#
#     def label_context(self, obs):
#         with torch.no_grad():
#             pi = self._distribution(obs)
#             con = pi.sample()
#             # print("Drawn context: ", con)
#             # logp_con = self._log_prob_from_distribution(pi, con)
#
#         # return logp_con.numpy()
#         return con
#
#
#
# ########
# def latent_loss(z_mean, z_stddev):
#     mean_sq = z_mean * z_mean
#     stddev_sq = z_stddev * z_stddev
#     return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)
#
#
# ########
