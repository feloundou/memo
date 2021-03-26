import time
import joblib
import os
import os.path as osp
import torch
from memo.utils.utils import *
import gym
import safety_gym
from cpprb import ReplayBuffer
import pandas as pd
from random import randint
import pickle
import wandb
import numpy as np
# from safety_gym.envs.engine import Engine



    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()

    if record:
        print("saving final buffer")
        bufname_pk = data_path + config_name + '_episodes/sim_data_' + str(int(num_episodes)) + '_buffer.pkl'
        file_pi = open(bufname_pk, 'wb')
        pickle.dump(rb.get_all_transitions(), file_pi)
        wandb.finish()

        return rb

    if benchmark:
        return ep_rewards, ep_costs


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--fpath', type=str,
                        default= '/home/tyna/Documents/openai/research-memo/data/ppo_test/ppo_test_s0/')
    parser.add_argument('--len', '-l', type=int, default=0)
    parser.add_argument('--episodes', '-n', type=int, default=100)
    parser.add_argument('--norender', '-nr', action='store_true')
    # parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    args = parser.parse_args()

    # the safe trained file is ppo_500e_8hz_cost5_rew1_lim25

    # file_name = 'ppo_500e_8hz_cost5_rew1_lim25'
    file_name = 'ppo_penalized_test'  # second best
    # file_name = 'ppo_penalized_cyan_500ep_8000steps'   # best so far
    # file_name = 'cpo_500e_8hz_cost1_rew1_lim25'  # unconstrained
    config_name = 'cyan'
    # file_name = 'ppo_penalized_' + config_name + '_20Ks_1Ke_128x4'
    # file_name = 'ppo_penalized_cyan_20Ks_1Ke_128x4'


    base_path = '/home/tyna/Documents/safe-experts/data/'
    expert_path = '/home/tyna/Documents/openai/research-memo/expert_data/'


    _, get_action = load_policy_and_env(osp.join(base_path, file_name, file_name + '_s0/'),
    # '/home/tyna/Documents/openai/research-memo/data/ppo_500e_8hz_cost1_rew1_lim25/ppo_500e_8hz_cost1_rew1_lim25_s0/',
                                        args.itr if args.itr >= 0 else 'last',
                                        args.deterministic)

    print("Get action: ", get_action)

    env = gym.make('Safexp-PointGoal1-v0')
    print("Just made env")
    env._seed = 0
    # run_policy(env, get_action, args.len, args.episodes, not (args.norender), record=False, data_path=base_path, config_name='cyan', max_len_rb)

    # run_policy(env, get_action, args.len, args.episodes, False, record=True, data_path=expert_path, config_name=config_name, max_len_rb=10000)
    run_policy(env, get_action, args.len, args.episodes, True, record=True, data_path=expert_path,
               config_name=config_name, max_len_rb=10000)



