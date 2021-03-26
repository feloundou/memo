import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

from torch.nn import Parameter

import torch.nn.functional as F

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

from memo.utils.utils import *


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
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
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
        # print("test list")
        # print(list(hidden_sizes))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
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

    # def __init__(self, observation_space, action_space,
    #              hidden_sizes=(64, 64), activation=nn.LeakyReLU):
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
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
            vc = self.vc(obs)
            # pen = self.pen(obs)

        return a.numpy(), v.numpy(), vc.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]



class GaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        # log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        # self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        # self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)
        self.shared_net = mlp([obs_dim] + list(hidden_sizes), activation)
        self.mu_net = nn.Linear(hidden_sizes[-1], act_dim)
        self.var_net = nn.Linear(hidden_sizes[-1], act_dim)

    def forward(self, x):
        mu = self.mu_net(F.leaky_relu(self.shared_net(x)))
        std = self.var_net(F.leaky_relu(self.shared_net(x)))
        return Normal(loc=mu, scale=std).rsample()



class DistilledGaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, n_experts):
        super().__init__()
        obs_dim_aug = obs_dim + n_experts
        self.shared_net = mlp([obs_dim_aug] + list(hidden_sizes), activation)

        self.mu_net = nn.Linear(hidden_sizes[-1], act_dim)
        self.var_net = nn.Linear(hidden_sizes[-1], act_dim)

    def forward(self, x):

        out = F.leaky_relu(self.shared_net(x))
        mu = self.mu_net(out)
        std = self.var_net(out)

        return Normal(loc=mu, scale=std).rsample()

class Discriminator(nn.Module):
    def __init__(self, obs_space, act_space, hidden_sizes, activation=nn.Tanh):
        super().__init__()
        obs_dim = obs_space.shape[0]
        act_dim = act_space.shape[0]
        discrim_dim = obs_dim + act_dim
        self.discrim_net = mlp([discrim_dim] + list(hidden_sizes) + [1], activation)


    def forward(self, obs):
        prob = torch.sigmoid(self.discrim_net(obs))
        return prob


class VDB(nn.Module):
    def __init__(self, obs_space, act_space, hidden_sizes, activation=nn.Tanh):
        super(VDB, self).__init__()
        obs_dim = obs_space.shape[0]
        act_dim = act_space.shape[0]
        discrim_dim = obs_dim + act_dim
        z_size = 128

        # self.fc1 = nn.Linear(num_inputs, args.hidden_size)
        self.fc1 = nn.Linear(discrim_dim, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], z_size)
        self.fc3 = nn.Linear(hidden_sizes[0], z_size)
        self.fc4 = nn.Linear(z_size, hidden_sizes[0])
        self.fc5 = nn.Linear(hidden_sizes[0], 1)

        self.fc5.weight.data.mul_(0.1)
        self.fc5.bias.data.mul_(0.0)

    def encoder(self, x):
        h = torch.tanh(self.fc1(x))
        return self.fc2(h), self.fc3(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar / 2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def discriminator(self, z):
        h = torch.tanh(self.fc4(z))
        return torch.sigmoid(self.fc5(h))

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        prob = self.discriminator(z)
        return prob, mu, logvar



def fc_q(env, hidden1=400, hidden2=300):
    return nn.Sequential(
        nn.Linear(env.state_space.shape[0] +
                  env.action_space.shape[0], hidden1),
        nn.LeakyReLU(),
        nn.Linear(hidden1, hidden2),
        nn.LeakyReLU(),
        nn.Linear(hidden2, 1),
    )


def fc_v(env, hidden1=400, hidden2=300):
    return nn.Sequential(
        nn.Linear(env.state_space.shape[0], hidden1),
        nn.LeakyReLU(),
        nn.Linear(hidden1, hidden2),
        nn.LeakyReLU(),
        nn.Linear(hidden2, 1),
    )


def fc_deterministic_policy(env, hidden1=400, hidden2=300):
    return nn.Sequential(
        nn.Linear(env.state_space.shape[0], hidden1),
        nn.LeakyReLU(),
        nn.Linear(hidden1, hidden2),
        nn.LeakyReLU(),
        nn.Linear(hidden2, env.action_space.shape[0]),
    )


def fc_deterministic_noisy_policy(env, hidden1=400, hidden2=300):
    return nn.Sequential(
        nn.NoisyFactorizedLinear(env.state_space.shape[0], hidden1),
        nn.LeakyReLU(),
        nn.NoisyFactorizedLinear(hidden1, hidden2),
        nn.LeakyReLU(),
        nn.NoisyFactorizedLinear(hidden2, env.action_space.shape[0]),
    )


def fc_soft_policy(env, hidden1=400, hidden2=300):
    return nn.Sequential(
        nn.Linear(env.state_space.shape[0], hidden1),
        nn.LeakyReLU(),
        nn.Linear(hidden1, hidden2),
        nn.LeakyReLU(),
        nn.Linear(hidden2, env.action_space.shape[0] * 2),
    )


def fc_actor_critic(env, hidden1=400, hidden2=300):
    features = nn.Sequential(
        nn.Linear(env.state_space.shape[0], hidden1),
        nn.LeakyReLU(),
    )

    v = nn.Sequential(
        nn.Linear(hidden1, hidden2),
        nn.LeakyReLU(),
        nn.Linear(hidden2, 1)
    )

    policy = nn.Sequential(
        nn.Linear(hidden1, hidden2),
        nn.LeakyReLU(),
        nn.Linear(hidden2, env.action_space.shape[0] * 2)
    )

    return features, v, policy


def fc_discriminator(env, hidden1=400, hidden2=300):
    return nn.Sequential(
        nn.Linear(env.state_space.shape[0] + env.action_space.shape[0],
                  hidden1),
        nn.LeakyReLU(),
        nn.Linear(hidden1, hidden2),
        nn.LeakyReLU(),
        nn.Linear(hidden2, 1),
        nn.Sigmoid())


def fc_reward(env, hidden1=400, hidden2=300):
    return nn.Sequential(
        nn.Linear(env.state_space.shape[0] +
                  env.action_space.shape[0], hidden1),
        nn.LeakyReLU(),
        nn.Linear(hidden1, hidden2),
        nn.LeakyReLU(),
        nn.Linear(hidden2, 1)
    )


# parser.add_argument('--env_name', type=str, default="Hopper-v2",
#                     help='name of the environment to run')
# parser.add_argument('--load_model', type=str, default=None,
#                     help='path to load the saved model')
# parser.add_argument('--gamma', type=float, default=0.99,
#                     help='discounted factor (default: 0.99)')
# parser.add_argument('--lamda', type=float, default=0.98,
#                     help='GAE hyper-parameter (default: 0.98)')
# parser.add_argument('--hidden_size', type=int, default=100,
#                     help='hidden unit size of actor, critic and vdb networks (default: 100)')
# parser.add_argument('--z_size', type=int, default=4,
#                     help='latent vector z unit size of vdb networks (default: 4)')
# parser.add_argument('--learning_rate', type=float, default=3e-4,
#                     help='learning rate of models (default: 3e-4)')
# parser.add_argument('--l2_rate', type=float, default=1e-3,
#                     help='l2 regularizer coefficient (default: 1e-3)')
# parser.add_argument('--clip_param', type=float, default=0.2,
#                     help='clipping parameter for PPO (default: 0.2)')
# parser.add_argument('--alpha_beta', type=float, default=1e-4,
#                     help='step size to be used in beta term (default: 1e-4)')
# parser.add_argument('--i_c', type=float, default=0.5,
#                     help='constraint for KL-Divergence upper bound (default: 0.5)')
# parser.add_argument('--vdb_update_num', type=int, default=3,
#                     help='update number of variational discriminator bottleneck (default: 3)')
# parser.add_argument('--ppo_update_num', type=int, default=10,
#                     help='update number of actor-critic (default: 10)')
# parser.add_argument('--total_sample_size', type=int, default=2048,
#                     help='total sample size to collect before PPO update (default: 2048)')
# parser.add_argument('--batch_size', type=int, default=64,
#                     help='batch size to update (default: 64)')
# parser.add_argument('--suspend_accu_exp', type=float, default=0.8,
#                     help='accuracy for suspending discriminator about expert data (default: 0.8)')
# parser.add_argument('--suspend_accu_gen', type=float, default=0.8,
#                     help='accuracy for suspending discriminator about generated data (default: 0.8)')
# parser.add_argument('--max_iter_num', type=int, default=4000,
#                     help='maximal number of main iterations (default: 4000)')
# parser.add_argument('--seed', type=int, default=500,
#                     help='random seed (default: 500)')
# parser.add_argument('--logdir', type=str, default='logs',
#                     help='tensorboardx logs directory')
# args = parser.parse_args()