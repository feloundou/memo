# import time
# import joblib
# import os
# import os.path as osp
# import torch
# from memo.utils.utils import *
# import gym
# import safety_gym
# from safety_gym.envs.engine import Engine
#
# from memo.utils.torch_cpo_utils import *
# # from cpo_torch import CPO
# from train_expert_cpo import *
# from buffer_torch import *
# from models_torch import MLP_DiagGaussianPolicy, MLP
#
#
#
# def load_policy_and_env(fpath, itr='last', deterministic=False):
#     """
#     Load a policy from save along with RL env.
#     Not exceptionally future-proof, but it will suffice for basic uses of the
#     Spinning Up implementations.
#     loads as if there's a PyTorch save.
#     """
#
#     # handle which epoch to load from
#     if itr == 'last':
#         # check filenames for epoch (AKA iteration) numbers, find maximum value
#
#         pytsave_path = osp.join(fpath, 'pyt_save')
#         # Each file in this folder has naming convention 'modelXX.pt', where
#         # 'XX' is either an integer or empty string. Empty string case
#         # corresponds to len(x)==8, hence that case is excluded.
#         saves = [int(x.split('.')[0][5:]) for x in os.listdir(pytsave_path) if len(x) > 8 and 'model' in x]
#
#         itr = '%d' % max(saves) if len(saves) > 0 else ''
#
#     else:
#         assert isinstance(itr, int), \
#             "Bad value provided for itr (needs to be int or 'last')."
#         itr = '%d' % itr
#
#     # load the get_action function
#     get_action = load_pytorch_policy(fpath, itr, deterministic)
#
#     # try to load environment from save
#     # (sometimes this will fail because the environment could not be pickled)
#     try:
#         print("path")
#         print(osp.join(fpath, 'vars' + itr + '.pkl'))
#         state = joblib.load(osp.join(fpath, 'vars' + itr + '.pkl'))
#         print("test1")
#         print(state)
#         env = state['env']
#     except:
#         env = None
#
#     return env, get_action
#
#
# def load_pytorch_policy(fpath, itr, deterministic=False):
#     """ Load a pytorch policy saved with Spinning Up Logger."""
#
#     fname = osp.join(fpath, 'pyt_save', 'model' + itr + '.pt')
#     print('\n\nLoading from %s.\n\n' % fname)
#
#     model = torch.load(fname)
#
#     # make function for producing an action given a single state
#     def get_action(x):
#         with torch.no_grad():
#             x = torch.as_tensor(x, dtype=torch.float32)
#             action = model.act(x)
#         return action
#
#     def get_action_cpo(x):
#         with torch.no_grad():
#             x = torch.as_tensor(x, dtype=torch.float32)
#             action = model.act(x)
#         return action
#
#     return get_action
#
#
# def run_policy(env, get_action, max_ep_len=None, num_episodes=100, render=True):
#     assert env is not None, \
#         "Environment not found!\n\n It looks like the environment wasn't saved, " + \
#         "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
#         "page on Experiment Outputs for how to handle this situation."
#
#     logger = EpochLogger()
#     o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
#     while n < num_episodes:
#         if render:
#             env.render()
#             time.sleep(1e-3)
#
#         a = get_action(o)
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
# if __name__ == '__main__':
#     import argparse
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--fpath', type=str,
#                         default= '/home/tyna/Documents/openai/research-memo/data/ppo_test/ppo_test_s0/')
#     parser.add_argument('--env_name', type=str, default='Safexp-PointGoal1-v0')
#     parser.add_argument('--len', '-l', type=int, default=0)
#     parser.add_argument('--episodes', '-n', type=int, default=100)
#     parser.add_argument('--norender', '-nr', action='store_true')
#     # parser.add_argument('--itr', '-i', type=int, default=-1)
#     parser.add_argument('--itr', '-i', type=int, default=-1)
#     parser.add_argument('--deterministic', '-d', action='store_true')
#     args = parser.parse_args()
#
#
# file_name = 'cpo_500e_8hz_cost1_rew1_lim25'
# base_path = '/home/tyna/Documents/openai/research-memo/data/'
#
# env = gym.make(args.env_name)
# state_dim = env.observation_space.shape[0]
# action_dim = env.action_space.shape[0]
#
# epochs = 30
# n_episodes = 5
# # n_episodes = 10000
# max_ep_len = 16
# policy_dims = [64, 64]
# vf_dims = [64, 64]
# cf_dims = [64, 64]
# cost_lim = 10
#
# # Gaussian policy
# policy = MLP_DiagGaussianPolicy(state_dim, policy_dims, action_dim)
# value_fun = MLP(state_dim + 1, vf_dims, 1)
# cost_fun = MLP(state_dim + 1, cf_dims, 1)
#
# simulator = SinglePathSimulator(args.env_name, policy, n_episodes, max_ep_len)
#
# cpo = CPO(policy,
#           value_fun,
#           cost_fun,
#           simulator,
#           # model_name='cpo-test-new',
#           # cost_lim=args.cost_lim,
#           model_name = 'cpo-run-500e',
#           # 'Safexp-PointGoal1-v0',
#           continue_from_file=True,
#           save_dir = '/home/tyna/Documents/openai/research-memo/data/cpo-run-500e/pyt_save/'
#           # '/home/tyna/Documents/openai/research-memo/data/cpo_500e_8hz_cost1_rew1_lim25/cpo_500e_8hz_cost1_rew1_lim25_s0/pyt_save/'
#           )
#
#
#
# env = gym.make('Safexp-PointGoal1-v0')
# # run_policy(env, get_action, args.len, args.episodes, not (args.norender))
# buffer = simulator.run_sim(render=True)
# print("buffer has run")
#
# # import replay buffer object, populate with actions, rewards, dones, info
# # save it to a file, then you have a bunch of environment interactions from a given policy
#
# # store all intermediate action, data, similar to
# # >>> env = your_env.make()
# # >>> run_policy(env, get_action)
#
