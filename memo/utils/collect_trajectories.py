
from tqdm import tqdm
import os
import numpy as np
import torch
import os.path as osp

import gym
import safety_gym
from memo.simulations.run_policy_sim_ppo import load_policy_and_env


class SerializedBuffer:

    def __init__(self, path, device):
        tmp = torch.load(path)
        self.buffer_size = self._n = tmp['state'].size(0)
        self.device = device

        self.states = tmp['state'].clone().to(self.device)
        self.actions = tmp['action'].clone().to(self.device)
        self.rewards = tmp['reward'].clone().to(self.device)
        self.dones = tmp['done'].clone().to(self.device)
        self.next_states = tmp['next_state'].clone().to(self.device)

    def sample(self, batch_size):
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.next_states[idxes]
        )


class Buffer(SerializedBuffer):

    def __init__(self, buffer_size, state_shape, action_shape):
        self._n = 0
        self._p = 0
        self.buffer_size = buffer_size
        # self.device = device

        self.states = torch.empty(
            (buffer_size, *state_shape), dtype=torch.float)
        self.actions = torch.empty(
            (buffer_size, *action_shape), dtype=torch.float)
        self.rewards = torch.empty(
            (buffer_size, 1), dtype=torch.float)
        self.dones = torch.empty(
            (buffer_size, 1), dtype=torch.float)
        self.next_states = torch.empty(
            (buffer_size, *state_shape), dtype=torch.float)

    def append(self, state, action, reward, done, next_state):
        self.states[self._p].copy_(torch.from_numpy(state))
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.rewards[self._p] = float(reward)
        self.dones[self._p] = float(done)
        self.next_states[self._p].copy_(torch.from_numpy(next_state))

        self._p = (self._p + 1) % self.buffer_size
        self._n = min(self._n + 1, self.buffer_size)

    def save(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        print("path")
        print(path)

        torch.save({
            'state': self.states.clone().cpu(),
            'action': self.actions.clone().cpu(),
            'reward': self.rewards.clone().cpu(),
            'done': self.dones.clone().cpu(),
            'next_state': self.next_states.clone().cpu(),
        }, path)


def soft_update(target, source, tau):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.mul_(1.0 - tau)
        t.data.add_(tau * s.data)


def disable_gradient(network):
    for param in network.parameters():
        param.requires_grad = False


def add_random_noise(action, std):
    action += np.random.randn(*action.shape) * std
    return action.clip(-1.0, 1.0)


def collect_demo(env, buffer_size,
                 std, p_rand, seed=0):
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    config_name = 'marigold'
    file_name = 'ppo_penalized_' + config_name + '_128x4'

    base_path = '/home/tyna/Documents/openai/research-memo/data/'
    # expert_path = '/home/tyna/Documents/openai/research-memo/expert_data/'

    _, get_action = load_policy_and_env(osp.join(base_path, file_name, file_name + '_s0/'),
                                         'last',
                                        deterministic=False)

    print("wow")

    buffer = Buffer(
        buffer_size=buffer_size,
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        # device=device
    )

    total_return = 0.0
    num_episodes = 0

    state = env.reset()
    t = 0
    episode_return = 0.0

    for _ in tqdm(range(1, buffer_size + 1)):
        t += 1

        if np.random.rand() < p_rand:
            action = env.action_space.sample()
        else:
            # action = algo.exploit(state)
            # action = add_random_noise(action, std)
            action = get_action(state)

        next_state, reward, done, _ = env.step(action)
        max_episode_steps = 1000
        mask = False if t == max_episode_steps else done
        buffer.append(state, action, reward, mask, next_state)
        episode_return += reward

        if done:
            num_episodes += 1
            total_return += episode_return
            state = env.reset()
            t = 0
            episode_return = 0.0

        state = next_state

    print(f'Mean return of the expert is {total_return / num_episodes}')
    return buffer


def run(envName):
    env = gym.make(envName)

    # buffer_size = 10**6
    buffer_size = 10 ** 5

    buffer = collect_demo(
        env=env,
        buffer_size= buffer_size,
        std=0,
        p_rand=0,
        seed=0
    )
    buffer.save(osp.join(
        'buffers',
        'marigold',
        f'size{buffer_size}.pt'
    ))



envName = 'Safexp-PointGoal1-v0'
env = gym.make(envName)

run(envName)

