import torch
from functools import wraps
from inspect import getfullargspec, isfunction
from itertools import starmap

import gym
import numpy as np

from memo.utils.buffer_utils import Trajectory, Buffer
import os.path as osp

DEFAULT_DATA_DIR = osp.join(osp.abspath(osp.dirname(osp.dirname(__file__))),'data')

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

class Simulator:
    @autoassign(exclude=('env_name', 'env_args'))
    def __init__(self, env_name, policy, n_episodes, max_ep_len, obs_filter=None, **env_args):
        print("simulator init")
        self.env = np.asarray([gym.make(env_name) for i in range(n_episodes)])
        self.n_trajectories = n_episodes

        for env in self.env:
            env._max_episode_steps = max_ep_len


class ExpertSinglePathSimulator:
    def __init__(self, env_name, policy, n_trajectories, trajectory_len,
                 state_filter=None,
                 **env_args):
        Simulator.__init__(self, env_name, policy, n_trajectories, trajectory_len,
                           state_filter,  **env_args)

    def run_sim(self, sampling_mode=True, render=False, seed=None, mode="", demo_pi=[1,0]):
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

                if mode != "demo":
                    action_dists = self.policy(policy_input)

                    if sampling_mode:
                        actions = action_dists.sample()
                        actions = actions.cpu()
                    else:
                        actions = torch.Tensor(action_dists)
                else:
                    actions = torch.Tensor(demo_pi)
                # print("actions sampled: ", actions)


                for env, action, trajectory in zip(continuing_envs, actions, trajs_to_update):
                    if mode == "demo":
                        action = torch.as_tensor(demo_pi)
                    traj_count += 1

                    # obs, reward, trajectory.done, info = env.step(action.numpy())  # new change
                    obs, reward, trajectory.done, info = env.step(action)
                    # print("action taken: ", action)

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