# import scipy.signal
from gym.spaces import Box, Discrete
import numpy as np
import torch
from torch import nn
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
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

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
            print("pi dist! ", pi)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
            vc = self.vc(obs)

        return a.numpy(), v.numpy(), vc.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]


class MEMOActor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(MEMOActor, self).__init__()
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
