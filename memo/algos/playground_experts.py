import time
import joblib, os
import os.path as osp
import gym
import safety_gym
import pandas as pd
from random import randint
import pickle
import torch
import numpy as np
# from safety_gym.envs.engine import Engine
from memo.utils.utils import EpochLogger, num_procs
from memo.utils.utils import load_policy_and_env
from memo.algos.demo_policies import spinning_top_policy, circle_policy, square_policy, \
    forward_policy, back_forth_policy, forward_spin_policy

def run_demo_policy(env, demo_policy, max_ep_len=None, num_episodes=100, render=True, record=False, record_project='benchmarking', record_name='trained', data_path='', config_name='test', max_len_rb=100, benchmark=False, log_prefix=''):
    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
        "page on Experiment Outputs for how to handle this situation."

    logger = EpochLogger()
    o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
    ep_cost = 0
    step=0
    local_steps_per_epoch = int(4000 / num_procs())

    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    while n < num_episodes:
        if render:
            env.render()
            time.sleep(1e-3)

        a = get_action(o)
        # a = demo_policy[step]

        print("action taken: ", a)
        next_o, r, d, info = env.step(a)
        step += 1

        ep_ret += r
        ep_len += 1
        ep_cost += info['cost']

        # Important!
        o = next_o

        if d or (ep_len == max_ep_len): # finish recording and save csv
            logger.store(EpRet=ep_ret, EpLen=ep_len, EpCost=ep_cost)
            print('Episode %d \t EpRet %.3f \t EpLen %d \t EpCost %d' % (n, ep_ret, ep_len, ep_cost))
            o, r, d, ep_ret, ep_len, ep_cost = env.reset(), 0, False, 0, 0, 0
            step = 0
            n += 1


    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--fpath', type=str, default='/home/tyna/Documents/openai/research-project/data/ppo_test/ppo_test_s0/')
    parser.add_argument('--len', '-l', type=int, default=0)
    parser.add_argument('--episodes', '-n', type=int, default=100)
    parser.add_argument('--norender', '-nr', action='store_true')
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    args = parser.parse_args()

    # file_name = 'square_policy'
    file_name = 'back_forth_policy'

    base_path = '/home/tyna/Documents/memo/memo/data/'

    _, get_action = load_policy_and_env(osp.join(base_path, file_name, file_name + '_s0/'),
                                        args.itr if args.itr >= 0 else 'last',
                                        args.deterministic)

    env = gym.make('Safexp-PointGoal1-v0')
    print("Just made env")
    env._seed = 0

    run_demo_policy(env, square_policy, args.len, args.episodes, True)
