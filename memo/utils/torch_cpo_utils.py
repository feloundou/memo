import torch
# from torch.autograd import grad
#
import os as osp
#
# from torch.distributions import Independent
# from torch.distributions.categorical import Categorical
# from torch.distributions.kl import kl_divergence
# from torch.distributions.normal import Normal
#
from functools import wraps
from inspect import getfullargspec, isfunction
from itertools import starmap
#
import gym
# import safety_gym
import numpy as np
# import time
#
from memo.utils.buffer_torch import Trajectory, Buffer
import os.path as osp

DEFAULT_DATA_DIR = osp.join(osp.abspath(osp.dirname(osp.dirname(__file__))),'data')
#
# def cg_solver(Avp_fun, b, max_iter=10):
#     '''
#     Finds an approximate solution to a set of linear equations Ax = b
#     Parameters
#     ----------
#     Avp_fun : callable
#         a function that right multiplies a matrix A by a vector
#     b : torch.FloatTensor
#         the right hand term in the set of linear equations Ax = b
#     max_iter : int
#         the maximum number of iterations (default is 10)
#     Returns
#     -------
#     x : torch.FloatTensor
#         the approximate solution to the system of equations defined by Avp_fun
#         and b
#     '''
#
#     # device = get_device()
#     x = torch.zeros_like(b)
#         # .to(device)
#     r = b.clone()
#     p = b.clone()
#
#     for i in range(max_iter):
#         Avp = Avp_fun(p, retain_graph=True)
#
#         alpha = torch.matmul(r, r) / torch.matmul(p, Avp)
#         x += alpha * p
#
#         if i == max_iter - 1:
#             return x
#
#         r_new = r - alpha * Avp
#         beta = torch.matmul(r_new, r_new) / torch.matmul(r, r)
#         r = r_new
#         p = r + beta * p
#
#
#
#
# def get_Hvp_fun(functional_output, inputs, damping_coef=0.0):
#     '''
#     Returns a function that calculates a Hessian-vector product with the Hessian
#     of functional_output w.r.t. inputs
#     Parameters
#     ----------
#     functional_output : torch.FloatTensor (with requires_grad=True)
#         the output of the function of which the Hessian is calculated
#     inputs : torch.FloatTensor
#         the inputs w.r.t. which the Hessian is calculated
#     damping_coef : float
#         the multiple of the identity matrix to be added to the Hessian
#     '''
#
#     inputs = list(inputs)
#     grad_f = flat_grad(functional_output, inputs, create_graph=True)
#
#     def Hvp_fun(v, retain_graph=True):
#         gvp = torch.matmul(grad_f, v)
#         Hvp = flat_grad(gvp, inputs, retain_graph=retain_graph)
#         Hvp += damping_coef * v
#
#         return Hvp
#
#     return Hvp_fun
#
#
# def line_search(search_dir, max_step_len, constraints_satisfied, line_search_coef=0.9,
#                 max_iter=10):
#     '''
#     Perform a backtracking line search that terminates when constraints_satisfied
#     return True and return the calculated step length. Return 0.0 if no step
#     length can be found for which constraints_satisfied returns True
#     Parameters
#     ----------
#     search_dir : torch.FloatTensor
#         the search direction along which the line search is done
#     max_step_len : torch.FloatTensor
#         the maximum step length to consider in the line search
#     constraints_satisfied : callable
#         a function that returns a boolean indicating whether the constraints
#         are met by the current step length
#     line_search_coef : float
#         the proportion by which to reduce the step length after each iteration
#     max_iter : int
#         the maximum number of backtracks to do before return 0.0
#     Returns
#     -------
#     the maximum step length coefficient for which constraints_satisfied evaluates
#     to True
#     '''
#
#     step_len = max_step_len / line_search_coef
#
#     for i in range(max_iter):
#         step_len *= line_search_coef
#
#         if constraints_satisfied(step_len * search_dir, step_len):
#             return step_len
#
#     return torch.tensor(0.0)
#
#
#
#
# def detach_dist(dist):
#     '''
#     Return a copy of dist with the distribution parameters detached from the
#     computational graph
#     Parameters
#     ----------
#     dist: torch.distributions.distribution.Distribution
#         the distribution object for which the detached copy is to be returned
#     Returns
#     -------
#     detached_dist
#         the detached distribution
#     '''
#
#     if type(dist) is Categorical:
#         detached_dist = Categorical(logits=dist.logits.detach())
#     elif type(dist) is Independent:
#         detached_dist = Normal(loc=dist.mean.detach(), scale=dist.stddev.detach())
#         detached_dist = Independent(detached_dist, 1)
#
#     return detached_dist
#
# def mean_kl_first_fixed(dist_1, dist_2):
#     '''
#     Calculate the kl-divergence between dist_1 and dist_2 after detaching dist_1
#     from the computational graph
#     Parameters
#     ----------
#     dist_1 : torch.distributions.distribution.Distribution
#         the first argument to the kl-divergence function (will be fixed)
#     dist_2 : torch.distributions.distribution.Distribution
#         the second argument to the kl-divergence function (will not be fixed)
#     Returns
#     -------
#     mean_kl : torch.float
#         the kl-divergence between dist_1 and dist_2
#     '''
#     dist_1_detached = detach_dist(dist_1)
#     mean_kl = torch.mean(kl_divergence(dist_1_detached, dist_2))
#
#     return mean_kl
#
#
#
# def get_device():
#     '''
#     Return a torch.device object. Returns a CUDA device if it is available and
#     a CPU device otherwise.
#     '''
#     if torch.cuda.is_available():
#         return torch.device('cuda')
#     else:
#         return torch.device('cpu')
#
# # save_dir = 'saved-sessions'
#
# def set_params(parameterized_fun, new_params):
#     '''
#     Set the parameters of parameterized_fun to new_params
#     Parameters
#     ----------
#     parameterized_fun : torch.nn.Sequential
#         the function approximator to be updated
#     update : torch.FloatTensor
#         a flattened version of the parameters to be set
#     '''
#
#     n = 0
#
#     for param in parameterized_fun.parameters():
#         numel = param.numel()
#         new_param = new_params[n:n + numel].view(param.size())
#         param.data = new_param
#         n += numel
#
# def flatten(vecs):
#     '''
#     Return an unrolled, concatenated copy of vecs
#     Parameters
#     ----------
#     vecs : list
#         a list of Pytorch Tensor objects
#     Returns
#     -------
#     flattened : torch.FloatTensor
#         the flattened version of vecs
#     '''
#
#     flattened = torch.cat([v.view(-1) for v in vecs])
#
#     return flattened
#
# def flat_grad(functional_output, inputs, retain_graph=False, create_graph=False):
#     '''
#     Return a flattened view of the gradients of functional_output w.r.t. inputs
#     Parameters
#     ----------
#     functional_output : torch.FloatTensor
#         The output of the function for which the gradient is to be calculated
#     inputs : torch.FloatTensor (with requires_grad=True)
#         the variables w.r.t. which the gradient will be computed
#     retain_graph : bool
#         whether to keep the computational graph in memory after computing the
#         gradient (not required if create_graph is True)
#     create_graph : bool
#         whether to create a computational graph of the gradient computation
#         itself
#     Return
#     ------
#     flat_grads : torch.FloatTensor
#         a flattened view of the gradients of functional_output w.r.t. inputs
#     '''
#
#     if create_graph == True:
#         retain_graph = True
#
#     grads = grad(functional_output, inputs, retain_graph=retain_graph, create_graph=create_graph)
#     flat_grads = flatten(grads)
#
#     return flat_grads
#
# def get_flat_params(parameterized_fun):
#     '''
#     Get a flattened view of the parameters of a function approximator
#     Parameters
#     ----------
#     parameterized_fun : torch.nn.Sequential
#         the function approximator for which the parameters are to be returned
#     Returns
#     -------
#     flat_params : torch.FloatTensor
#         a flattened view of the parameters of parameterized_fun
#     '''
#     parameters = parameterized_fun.parameters()
#     flat_params = flatten([param.view(-1) for param in parameters])
#
#     return flat_params
#
# def normalize(x):
#     mean = torch.mean(x)
#     std = torch.std(x)
#     x_norm = (x - mean) / std
#
#     return x_norm
#
#

def autoassign(*names, **kwargs):
    """
    autoassign(function) -> method
    autoassign(*argnames) -> decorator
    autoassign(exclude=argnames) -> decorator
    allow a method to assign (some of) its arguments as attributes of
    'self' automatically.  E.g.
    >>> class Foo(object):
    ...     @autoassign
    ...     def __init__(self, foo, bar): pass
    ...
    >>> breakfast = Foo('spam', 'eggs')
    >>> breakfast.foo, breakfast.bar
    ('spam', 'eggs')
    To restrict autoassignment to 'bar' and 'baz', write:
        @autoassign('bar', 'baz')
        def method(self, foo, bar, baz): ...
    To prevent 'foo' and 'baz' from being autoassigned, use:
        @autoassign(exclude=('foo', 'baz'))
        def method(self, foo, bar, baz): ...
    """
    if kwargs:
        exclude, f = set(kwargs['exclude']), None
        sieve = lambda l:filter(lambda nv: nv[0] not in exclude, l)
    elif len(names) == 1 and isfunction(names[0]):
        f = names[0]
        sieve = lambda l:l
    else:
        names, f = set(names), None
        sieve = lambda l: filter(lambda nv: nv[0] in names, l)
    def decorator(f):
        fargnames, _, _, fdefaults, _, _, _ = getfullargspec(f)
        # Remove self from fargnames and make sure fdefault is a tuple
        fargnames, fdefaults = fargnames[1:], fdefaults or ()
        defaults = list(sieve(zip(reversed(fargnames), reversed(fdefaults))))
        @wraps(f)
        def decorated(self, *args, **kwargs):
            assigned = dict(sieve(zip(fargnames, args)))
            assigned.update(sieve(kwargs.items()))
            for _ in starmap(assigned.setdefault, defaults): pass
            self.__dict__.update(assigned)
            return f(self, *args, **kwargs)
        return decorated
    return f and decorator(f) or decorator


#
# class RunningStat:
#     '''
#     Keeps track of a running estimate of the mean and standard deviation of
#     a distribution based on the observations seen so far
#     Attributes
#     ----------
#     _M : torch.float
#         estimate of the mean of the observations seen so far
#     _S : torch.float
#         estimate of the sum of the squared deviations from the mean of the
#         observations seen so far
#     n : int
#         the number of observations seen so far
#     Methods
#     -------
#     update(x)
#         update the running estimates of the mean and standard deviation
#     mean()
#         return the estimated mean
#     var()
#         return the estimated variance
#     std()
#         return the estimated standard deviation
#     '''
#
#     def __init__(self):
#         self._M = None
#         self._S = None
#         self.n = 0
#
#     def update(self, x):
#         self.n += 1
#
#         if self.n == 1:
#             self._M = x.clone()
#             self._S = torch.zeros_like(x)
#         else:
#             old_M = self._M.clone()
#             self._M = old_M + (x - old_M) / self.n
#             self._S = self._S + (x - old_M) * (x - self._M)
#
#     @property
#     def mean(self):
#         return self._M
#
#     @property
#     def var(self):
#         if self.n > 1:
#             var = self._S / (self.n - 1)
#         else:
#             var = torch.pow(self.mean, 2)
#
#         return var
#
#     @property
#     def std(self):
#         return torch.sqrt(self.var)
#
#
#
# # def make_env(env_name, **env_args):
# #     if env_name == 'ant_gather':
# #         return PointGather(**env_args)
# #     elif env_name == 'point_gather':
# #         return PointGatherEnv(**env_args)
# #     elif env_name == "Safexp-PointGoal1-v0":
# #         return gym.make(env_name)
# #     else:
# #         raise NotImplementedError
#

class Simulator:
    @autoassign(exclude=('env_name', 'env_args'))
    def __init__(self, env_name, policy, n_episodes, max_ep_len, obs_filter=None, **env_args):
        print("simulator init")

        # self.env = np.asarray([make_env(env_name, **env_args) for i in range(n_trajectories)])
        self.env = np.asarray([gym.make(env_name) for i in range(n_episodes)])
        # print("environment")
        # print(self.env)
        self.n_trajectories = n_episodes

        for env in self.env:
            env._max_episode_steps = max_ep_len

        # self.device = get_device()


class ExpertSinglePathSimulator:
    def __init__(self, env_name, policy, n_trajectories, trajectory_len,
                 state_filter=None,
                 **env_args):
        Simulator.__init__(self, env_name, policy, n_trajectories, trajectory_len,
                           state_filter,  **env_args)

    def run_sim(self, sampling_mode=True, render=False, seed=None):
        print("policy eval launch")
        if seed is not None:
            init_seed = seed

        if sampling_mode:
            self.policy.eval()

        with torch.no_grad():
            trajectories = np.asarray([Trajectory() for i in range(self.n_trajectories)])
            continue_mask = np.ones(self.n_trajectories)
            traj_count = 0


            for env, trajectory in zip(self.env, trajectories):
                # initialize with fixed seed if necessary
                if seed is not None:
                    env._seed = init_seed
                obs = torch.tensor(env.reset()).float()

                # Maybe batch this operation later
                if self.obs_filter:
                    obs = self.obs_filter(obs)

                trajectory.observations.append(obs)
                old_obs=obs
                init_seed += 1

            while np.any(continue_mask):
                ep_cost, ep_reward = 0, 0
                continue_indices = np.where(continue_mask)
                trajs_to_update = trajectories[continue_indices]
                continuing_envs = self.env[continue_indices]

                policy_input = torch.stack([torch.tensor(trajectory.observations[-1])
                                            for trajectory in trajs_to_update])

                action_dists = self.policy(policy_input)

                if sampling_mode:
                    actions = action_dists.sample()
                    actions = actions.cpu()
                else:
                    actions = torch.Tensor(action_dists)
                # print("actions sampled: ", actions)


                for env, action, trajectory in zip(continuing_envs, actions, trajs_to_update):
                    traj_count += 1

                    obs, reward, trajectory.done, info = env.step(action.numpy())

                    obs = torch.tensor(obs).float()
                    reward = torch.tensor(reward, dtype=torch.float)
                    cost = torch.tensor(info['cost_hazards'], dtype=torch.float)

                    if self.obs_filter:
                        obs = self.obs_filter(obs)

                    trajectory.actions.append(action)
                    trajectory.rewards.append(reward)
                    trajectory.costs.append(cost)

                    new_obs = obs

                    obs_diff = new_obs - old_obs

                    if not trajectory.done:
                        trajectory.observations.append(obs)
                        trajectory.next_observations.append(new_obs)
                        trajectory.obs_diff.append(obs_diff)

                continue_mask = np.asarray([1 - trajectory.done for trajectory in trajectories])

        memory = Buffer(trajectories)
        print("Memory: ", memory)

        return memory, trajectories