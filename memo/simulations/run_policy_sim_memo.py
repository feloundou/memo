# import time
# import joblib
# import os
# import os.path as osp
# import torch
# from memo.utils.utils import EpochLogger
# import gym
# import safety_gym
# from gym.wrappers import Monitor
# import safety_gym
#
# from cpprb import ReplayBuffer
# import pickle
# import wandb
# import numpy as np
#
# from run_policy_sim_ppo import load_policy_and_env, load_pytorch_policy
#
#
# def run_memo_policy(env, get_action, max_ep_len=None, num_episodes=100, render=True):
#     assert env is not None, \
#         "Environment not found!\n\n It looks like the environment wasn't saved, " + \
#         "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
#         "page on Experiment Outputs for how to handle this situation."
#
#     logger = EpochLogger()
#     print("Inside one")
#     o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
#     while n < num_episodes:
#         if render:
#             env.render()
#             time.sleep(1e-3)
#
#         # a = get_action(o)
#         a = get_action(o, torch.as_tensor(0))
#         # print("GOT the action! ", a)
#         o, r, d, _ = env.step(a)
#         ep_ret += r
#         ep_len += 1
#
#         if d or (ep_len == max_ep_len):
#             logger.store(EpRet=ep_ret, EpLen=ep_len)
#             print('Episode %d \t EpRet %.3f \t EpLen %d' % (n, ep_ret, ep_len))
#             o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
#             n += 1
#
#     logger.log_tabular('EpRet', with_min_and_max=True)
#     logger.log_tabular('EpLen', average_only=True)
#     logger.dump_tabular()
#
#
#
# if __name__ == '__main__':
#     import argparse
#     import math
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--fpath', type=str,
#                         default= '/home/tyna/Documents/openai/research-memo/data/ppo_test/ppo_test_s0/')
#     parser.add_argument('--len', '-l', type=int, default=0)
#     parser.add_argument('--episodes', '-n', type=int, default=5)
#     parser.add_argument('--norender', '-nr', action='store_true')
#     parser.add_argument('--itr', '-i', type=int, default=-1)
#     parser.add_argument('--deterministic', '-d', action='store_true')
#     args = parser.parse_args()
#
#     # config_name= 'marigold'
#     # # file_name = 'memo-vae'
#     # file_name = 'ppo_penalized_marigold_128x4'
#     file_name = 'ppo_penalized_rose_128x4'
#     # file_name = 'memo-marigold-rose'
#     base_path = '/home/tyna/Documents/safe-experts/data/'
#
#     # _, get_action = load_policy_and_env(osp.join(base_path, file_name, file_name + '_s0/'), args.itr if args.itr >= 0 else 'last', args.deterministic, type='memo')
#     _, get_action = load_policy_and_env(osp.join(base_path, file_name, file_name + '_s0/'),
#                                         args.itr if args.itr >= 0 else 'last', args.deterministic, type='ppo')
#
#     # print("Get action: ", get_action)
#     env = gym.make('Safexp-PointGoal1-v0')
#     env = gym.wrappers.Monitor(env, "video", force=True)
#
#     # env.render()
#
#
#     theta = 0
#     factor = 2
#     render = True
#     n, num_episodes = 0, 1
#     A = [-1, -1]
#     i = 1
#
#     o = env.reset()
#
#     if render:
#         env.render()
#         time.sleep(1e-3)
#
#     while n < num_episodes:
#         env._seed = 0
#         # obs, rew, done, info = env.step(env.action_space.sample())
#         # a = get_action(o, torch.as_tensor(9))
#         # a = get_action(o)
#         # a = [factor*math.sin(theta), factor*math.cos(theta)]
#         # a = [x*i for x in A]
#         a = env.action_space.sample()
#         print("Action: ", a)
#
#         o, r, d, _ = env.step(a)
#
#         i += 1
#         theta += 1
#
#         if d:
#             break
#
#
#     # env = Monitor(gym.make('Safexp-PointGoal1-v0'), './video', force=True, uid="0", mode="evaluation")
#     # run_policy(env, get_action, args.len, args.episodes, not (args.norender), record=False, data_path=base_path, config_name='cyan', max_len_rb)
#
#     # run_memo_policy(env, get_action, args.len, args.episodes, True)
#
# # for the
#
