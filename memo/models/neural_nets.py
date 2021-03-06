# import scipy.signal
from gym.spaces import Box, Discrete
import numpy as np
import torch
from torch import nn
import IPython
# from torch.nn import Parameter
import torch.nn.functional as F
from torch.distributions import Independent, OneHotCategorical, Categorical
from torch.distributions.normal import Normal
# # from torch.distributions.categorical import Categorical

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
        self.log_std = nn.Parameter(torch.as_tensor(log_std))
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
            # print("pi dist! ", pi)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
            vc = self.vc(obs)

        return a.numpy(), v.numpy(), vc.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]



class MEMOActor(nn.Module):
    def __init__(self, state_dim, hidden_size, action_dim, activation=nn.Tanh):
        super(MEMOActor, self).__init__()

        log_std = -0.5 * np.ones(action_dim, dtype=np.float32)
        self.log_std = nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([state_dim] + hidden_size + [action_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def forward(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

# the critic is error here would be: reward + gamma*V(s_t+1)-V(s_t)
# http://incompleteideas.net/book/first/ebook/node66.html


class MEMO(nn.Module):
    """Multiple Experts, Multiple Objectives;
    """
    def __init__(self, obs_dim, out_dim, encoder_hidden, decoder_hidden, actor_hidden, latent_modes):
        '''
        :param obs_dim:
        :param latent_dim:
        :param out_dim:
        :param encoder_hidden:
        :param decoder_hidden:
        '''
        super(MEMO, self).__init__()
        self.found_contexts = []
        self.latent_modes = latent_modes

        self.num_embeddings = self.latent_modes
        self.embedding_dim = obs_dim
        self.vq_encoder = VQEncoder(obs_dim, self.embedding_dim)  # original

        self.prenet = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.vector_quantizer = VectorQuantizer(self.num_embeddings, self.embedding_dim)
        self.postnet = nn.Linear(self.embedding_dim, encoder_hidden[-1])
        self.vq_decoder = VQDecoder(encoder_hidden[-1], decoder_hidden, obs_dim)

        self.action_decoder = MEMOActor(state_dim=obs_dim + self.latent_modes, hidden_size=actor_hidden, action_dim=out_dim)
        # self.action_gaussian = GaussianActor(obs_dim=obs_dim + self.latent_modes, act_dim=out_dim,
        #                                      hidden_sizes=[128]*4, activation=nn.LeakyReLU)
        self.action_vq_dist = None


    def compute_quantized_loss(self, state, delta_state, actions):
        '''
        :param state:
        :param delta_state:
        :param actions:
        :return:
        '''
        delta_state_enc = self.vq_encoder(delta_state)   # In: [B, OBS_DIM]; Out: # [B, OBS_DIM]
        encoder_output = self.prenet(delta_state_enc)  # In: [B, OBS_DIM]; Out:  # [B, OBS_DIM]

        quantized, categorical_proposal, categorical_proposal_prob = self.vector_quantizer(encoder_output)

        # update the set of known contexts
        self.found_contexts = set([t.data.item() for t in categorical_proposal])

        # Straight Through Estimator (Some Magic)
        st_quantized = encoder_output + (quantized - encoder_output).detach()
        post_quantized = self.postnet(st_quantized)
        # print("Post Quantized: ", post_quantized)

        reconstruction = self.vq_decoder(post_quantized)
        # print("Reconstruction: ", reconstruction)

        categorical_proposal_reshape = torch.reshape(categorical_proposal, (-1, 1))
        categorical_proposal_onehot = F.one_hot(categorical_proposal_reshape, self.latent_modes).squeeze().float()
        # total_max = torch.tensor(0.)
        # print("distances max: ", max(total_max, torch.max(categorical_proposal_prob)))

        # concat_state_vq = torch.cat([state, categorical_proposal_onehot], dim=-1)
        concat_state_vq = torch.cat([state, categorical_proposal_prob], dim=-1)
        action_vq_dist = self.action_decoder(concat_state_vq)

        return encoder_output, quantized, reconstruction, categorical_proposal, action_vq_dist
        # return encoder_output, quantized, reconstruction, categorical_proposal, action_mse


    def act(self, state, context_label):
        concat_state_vq = torch.cat([state, torch.reshape(torch.as_tensor(context_label), (-1,))], dim=-1)
        action_vq_dist = self.action_decoder(concat_state_vq)
        action = action_vq_dist.sample()
        return action

    def forward(self, X, Delta_X, A, kl_beta=1., recon_gamma=1.):
        """
        Given input tensor, forward propagate, compute the loss, and backward propagate.
        Represents the lifecycle of a single iteration
        :param x: Raw state tensor
        :param Delta_x: State difference tensor
        :param a: Action tensor
        :param kl_beta: KL divergence temperance factor
        :param recon_gamma: State weights
        : Important to note that both recon and context loss cannot be negative.
        """
        encoder_output, quantized, reconstruction, vq_latent_labels, action_vq_dist =\
            self.compute_quantized_loss(X, Delta_X, A)

        vq_criterion = VQCriterion(beta=kl_beta)
        vq_total_loss, recons_loss, vq_loss, commitment_loss = vq_criterion(Delta_X, encoder_output, quantized, reconstruction)

        # original formula
        loss_pi = (torch.tensor(1.)/(torch.exp(action_vq_dist.log_prob(A)) + torch.tensor(0.1))).sum(axis=-1)
        loss = loss_pi * vq_total_loss

        return loss, loss_pi, X, vq_latent_labels, vq_total_loss




class VQEncoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(VQEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim // 2),
            nn.Tanh(),
            nn.Linear(out_dim // 2, out_dim),
            nn.Tanh()
        )

        # self.net = nn.Sequential(
        #     nn.Linear(in_dim, out_dim),
        #     nn.Tanh(),
        #     nn.Linear(out_dim, out_dim),
        #     nn.Tanh(),
        #     nn.Linear(out_dim, out_dim),
        #     nn.Tanh()
        # )

    def forward(self, input):
        return self.net(input)


class Clamper(nn.Module):
    def __init__(self, min=None, max=None):
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, input):
        return torch.clamp(input, self.min, self.max)

class VQDecoder(nn.Module):
    def __init__(self, obs_dim, hidden_dim, out_dim, activation=nn.Tanh):
        super().__init__()
        self.initial_act = nn.Tanh()
        self.net = mlp([obs_dim] + hidden_dim + [out_dim], activation)
    def forward(self, input):
        return self.net(self.initial_act(input))


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings  # E_N
        self.embedding_dim = embedding_dim    # E_D
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)

        self.scale = 1. / self.num_embeddings  # decimal
        print("Quantizer Scale: ", self.scale)
        nn.init.uniform_(self.embeddings.weight, -self.scale, self.scale)

    def proposal_distribution(self, input):
        input_shape = input.shape  # [B, OBS_DIM]
        flatten_input = input.flatten(end_dim=-2).contiguous()  # [B, OBS_DIM]
        distances = (flatten_input ** 2).sum(dim=1, keepdim=True) # [B, 1]
        distances = distances + (self.embeddings.weight ** 2).sum(dim=1) # [B, E_N]
        distances -= 2 * flatten_input @ self.embeddings.weight.t()  # [B, E_N]

        categorical_posterior = torch.argmin(distances, dim=-1) # [B]  # original
        categorical_posterior_prob = distances
        # categorical_posterior_prob = torch.clamp(distances, 0, 10)  # 10 is a hyperparameter
        # categorical_posterior_prob = torch.clamp(distances, 0, 5)  # 5 is a hyperparameter

        return categorical_posterior, categorical_posterior_prob

    def forward(self, input):
        proposal, proposal_prob = self.proposal_distribution(input) # [B]
        quantized = self.embeddings(proposal).contiguous() # [B, OBS_DIM]
        return quantized, proposal, proposal_prob


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

        return total_loss, reconstruction_loss, vq_loss, commitment_loss


class VDB(nn.Module):
    def __init__(self, num_inputs, args):
        super(VDB, self).__init__()
        self.fc1 = nn.Linear(num_inputs, args.hidden_size)
        self.fc2 = nn.Linear(args.hidden_size, args.z_size)
        self.fc3 = nn.Linear(args.hidden_size, args.z_size)
        self.fc4 = nn.Linear(args.z_size, args.hidden_size)
        self.fc5 = nn.Linear(args.hidden_size, 1)

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


###########################################################################3
from torch.autograd import Variable
from torch.distributions import Distribution, Normal


class TanhNormal(torch.distributions.Distribution):
    """
    Represent distribution of X where
        X ~ tanh(Z)
        Z ~ N(mean, std)
    Note: this is not very numerically stable.
    """
    def __init__(self, normal_mean, normal_std, epsilon=1e-6):
        """
        :param normal_mean: Mean of the normal distribution
        :param normal_std: Std of the normal distribution
        :param epsilon: Numerical stability epsilon when computing log-prob.
        """
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.normal = Normal(normal_mean, normal_std)
        self.epsilon = epsilon

    def sample_n(self, n, return_pre_tanh_value=False):
        z = self.normal.sample_n(n)
        if return_pre_tanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def log_prob(self, value, pre_tanh_value=None):
        """
        :param value: some value, x
        :param pre_tanh_value: arctanh(x)
        :return:
        """
        if pre_tanh_value is None:
            pre_tanh_value = torch.log(
                (1+value) / (1-value)
            ) / 2
        return self.normal.log_prob(pre_tanh_value) - torch.log(
            1 - value * value + self.epsilon
        )

    def sample(self, return_pretanh_value=False):
        z = self.normal.sample()
        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def rsample(self, return_pretanh_value=False):
        z = (
            self.normal_mean +
            self.normal_std *
            Variable(Normal(
                np.zeros(self.normal_mean.size()),
                np.ones(self.normal_std.size())
            ).sample())
        )
        # z.requires_grad_()
        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)


