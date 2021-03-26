# from abc import ABC, abstractmethod
# from cpprb import ReplayBuffer, create_before_add_func
# import wandb
# import numpy as np
#
# import os
# import os.path as osp
#
# import pickle
# import torch
# from memo.utils.utils import mpi_avg, EpochLogger, colorize, samples_from_cpprb, setup_pytorch_for_mpi, \
#     proc_id, setup_logger_kwargs, samples_to_np
# # from neural_nets import MLP
#
# from six.moves.collections_abc import Sequence
# from run_policy_sim_ppo import load_policy_and_env, run_policy
#
# # Representation utils
# def one_hot(a, num_classes):
#     return np.squeeze(np.eye(num_classes)[a.reshape(-1)])
#
#
#
# # Plot utils
# def line_series(xs, ys, keys=None, title=None, xname=None):
#     data = []
#     if not isinstance(xs[0], Sequence):
#         xs = [xs for _ in range(len(ys))]
#     assert len(xs) == len(ys), "Number of x-lines and y-lines must match"
#     for i, series in enumerate([list(zip(xs[i], ys[i])) for i in range(len(xs))]):
#         for x, y in series:
#             if keys is None:
#                 key = "key_{}".format(i)
#             else:
#                 key = keys[i]
#             data.append([x, key, y])
#
#     table = wandb.Table(data=data, columns=["step", "lineKey", "lineVal"])
#
#     return wandb.plot_table(
#         "wandb/lineseries/v0",
#         table,
#         {"step": "step", "lineKey": "lineKey", "lineVal": "lineVal"},
#         {"title": title, "xname": xname or "x"},
#     )
#
#
#
#
# # Abstract clone class
#
# class Clone(ABC):
#     """
#     Abstract clone class
#     """
#
#     @abstractmethod
#     def set_replay_buffer(self):
#         """
#         Set replay buffer from expert policies, passively
#         fetching or actively recording.
#         """
#
#     @abstractmethod
#     def train_clone(self):
#         """ Train a Clone according to its designated policy/
#         Args:
#             env: Str. Chosen Gym or Safety Gym environment
#             batch_size: Int. Size of sampling batches over the replay buffer.
#             train_iters: Int. Number of times to sample the replay buffer.
#             eval_episodes: Int. Number of episodes over which to evaluate the current clone policy.
#             eval_every: Int. Interval of epochs over which to evaluate the current clone policy.
#             eval_sample_efficiency: Bool. Whether or not to evaluate sample efficiency.
#             print_every: Int. Frequency at which to print loss output.
#             save_every: Int. Frequency at which to save clone policy.
#         Returns:
#             Trained clone
#
#         """
#         pass
#
#     @abstractmethod
#     def run_clone_sim(self):
#         "Run episodes from a pre-trained clone policy"
#         pass
#
#
# class BehavioralClone(Clone):
#     """
#     Clone class for Sampler.
#     """
#
#     def __init__(self,
#                  config_name,
#                  record_samples,
#                  clone_policy,
#                  optimizer,
#                  criterion,
#                  seed,
#                  expert_episodes,
#                  clone_epochs,
#                  replay_buffer_size,
#                  # evaluation=False,
#                  # store_samples=True
#                  ):
#         self.config_name = config_name
#         self.record_samples = record_samples
#         self.expert_episodes = expert_episodes
#         self.clone_epochs = clone_epochs
#         self.clone_policy = clone_policy
#         self.optimizer = optimizer
#         self.criterion = criterion
#         self.seed = seed
#         self.replay_buffer = None
#         self.replay_buffer_size = replay_buffer_size
#         self.table_name = 'Trained Clone ' + str(self.clone_epochs) + ' Epochs: '
#         self.expert_type = 'single'
#
#
#         self.max_steps = 1000
#
#         # Paths
#         self._memo_dir = '/home/tyna/Documents/openai/research-memo/'
#         self._root_data_path = self._memo_dir + 'data/'
#         self._expert_path = self._memo_dir + 'expert_data/'
#         self._clone_path = self._memo_dir + 'clone_data/'
#         self._demo_dir = os.path.join(self._expert_path, self.config_name + '_episodes/')
#         self._clone_dir = os.path.join(self._expert_path, self.config_name + '_episodes/')
#         self.file_name = 'ppo_penalized_' + self.config_name + '_128x4'
#         self.benchmark_memo_name = 'clone_benchmarking_' + self.config_name
#
#         # Special function to avoid certain slowdowns from PyTorch + MPI combo.
#         setup_pytorch_for_mpi()
#
#         # Random seed # seed = 0
#         self.seed += 10000 * proc_id()
#         torch.manual_seed(self.seed)
#         np.random.seed(self.seed)
#
#     def set_replay_buffer(self, env, get_from_file):
#
#         obs_dim = env.observation_space.shape
#         act_dim = env.action_space.shape
#
#         if get_from_file:
#             print(colorize("Pulling saved expert %s trajectories from file over %d episodes" %
#                            (self.config_name, self.expert_episodes), 'blue', bold=True))
#
#             f = open(self._demo_dir + 'sim_data_' + str(self.expert_episodes) + '_buffer.pkl', "rb")
#             buffer_file = pickle.load(f)
#             f.close()
#
#             data = samples_from_cpprb(npsamples=buffer_file)
#
#             # Reconstruct the data, then pass it to replay buffer
#             np_states, np_rewards, np_actions, np_next_states, np_dones, np_next_dones = samples_to_np(data)
#
#             # Create environment
#             before_add = create_before_add_func(env)
#
#             replay_buffer = ReplayBuffer(size= self.replay_buffer_size,
#                                          env_dict={
#                                              "obs": {"shape": obs_dim},
#                                              "act": {"shape": act_dim},
#                                              "rew": {},
#                                              "next_obs": {"shape": obs_dim},
#                                              "done": {}})
#
#             replay_buffer.add(**before_add(obs=np_states[~np_dones],
#                                            act=np_actions[~np_dones],
#                                            rew=np_rewards[~np_dones],
#                                            next_obs=np_next_states[~np_dones],
#                                            done=np_next_dones[~np_dones]))
#             self.replay_buffer = replay_buffer
#
#         else:
#             # Generate expert data
#             print(colorize(
#                 "Generating expert %s trajectories from file over %d episodes" % (self.config_name, self.expert_episodes),
#                 'blue', bold=True))
#
#             # Load trained policy
#             _, get_action = load_policy_and_env(osp.join(self._root_data_path, self.file_name, self.file_name + '_s0/'),
#                                                 'last', False)
#             expert_rb = run_policy(env,
#                                    get_action,
#                                    0,
#                                    self.expert_episodes,
#                                    False,
#                                    record=not get_from_file,
#                                    record_name='expert_' + self.file_name + '_' + str(self.expert_episodes) + '_runs',
#                                    record_memo='clone_benchmarking_' + self.config_name,
#                                    data_path= self._expert_path,
#                                    config_name= self.config_name,
#                                    max_len_rb=self.replay_buffer_size)
#
#             self.replay_buffer = expert_rb
#
#
#     def train_clone(self, env, batch_size, train_iters, eval_episodes, eval_every, eval_sample_efficiency, print_every, save_every):
#         # File Names
#
#         self.fname = self.config_name + "_clone_" + str(self.clone_epochs) + 'ep_' + str(train_iters) + 'trn' + '_' + \
#                      str(self.expert_episodes) + '_expert_runs'
#
#         # Load trained policy
#         _, expert_pi = load_policy_and_env(osp.join(self._root_data_path, self.file_name, self.file_name + '_s0/'), 'last', False)
#
#         # Set up logger and save configuration
#         logger_kwargs = setup_logger_kwargs(self.fname, self.seed, self._clone_path)
#         logger = EpochLogger(**logger_kwargs)
#         logger.setup_pytorch_saver(self.clone_policy)
#
#         print(colorize("Training clones of %s config over %s episodes" % (self.config_name, self.clone_epochs),
#                        'green', bold=True))
#
#         # Train without environment interaction
#         wandb.login()
#         wandb.init(memo=self.benchmark_memo_name, name=self.fname)
#
#         wandb.watch(self.clone_policy)  # watch neural net #only do this when not looping
#
#         AVG_R = []
#         AVG_C = []
#
#         tb_name = str(self.clone_epochs) + ' Epochs '
#
#         # Run the expert a few episodes (20) for a benchmark:
#         # Record 25 episodes from expert if we are evaluating for efficiency
#         if eval_sample_efficiency:
#             expert_rewards, expert_costs = run_policy(env,
#                                                       expert_pi,
#                                                       0,
#                                                       25,
#                                                       False,
#                                                       record=False,
#                                                       record_name='expert_' + self.file_name,
#                                                       record_memo='clone_benchmarking_' + self.config_name,
#                                                       data_path= self._expert_path,
#                                                       config_name=self.config_name,
#                                                       max_len_rb=  self.replay_buffer_size,
#                                                       benchmark=True,
#                                                       log_prefix=tb_name)
#
#         for epoch in range(self.clone_epochs):
#             total_loss = 0
#
#             for t in range(train_iters):
#                 # Sample from the replay buffer
#                 SAMPLE = self.replay_buffer.sample(batch_size)
#
#                 # Observe states and chosen actions from expert
#                 states = SAMPLE['obs']
#                 actions = SAMPLE['act']
#
#                 self.optimizer.zero_grad()
#
#                 # Policy loss
#                 a_pred = self.clone_policy(torch.tensor(states).float())
#                 loss = self.criterion(a_pred, torch.tensor(actions))
#
#                 # print("Loss!", loss)
#                 total_loss += loss.item()
#                 loss.backward()
#                 if t % print_every == print_every - 1:
#                     print(
#                         colorize('Epoch:%d Batch:%d Loss:%.4f' % (epoch, t + 1, total_loss / print_every), 'yellow',
#                                  bold=True))
#                     epoch_metrics = {'Avg Epoch Loss': total_loss / print_every}
#                     wandb.log(epoch_metrics)
#                     total_loss = 0
#
#                 self.optimizer.step()
#
#             if epoch % eval_every == eval_every - 1:
#                 if eval_sample_efficiency: # should we evaluate sample efficiency?
#
#                     avg_expert_rewards = np.mean(expert_rewards)
#                     avg_expert_costs = np.mean(expert_costs)
#
#                     return_list, cost_list = [], []
#
#                     for _ in range(eval_episodes):
#
#                         obs, done, steps, ep_reward, ep_cost = env.reset(), False, 0, 0, 0
#
#                         while not done:
#                             a = self.clone_policy(torch.tensor(obs).float())  # clone step
#                             obs, r, done, info = env.step(a.detach().numpy())
#                             cost = info['cost']
#
#                             ep_reward += r
#                             ep_cost += cost
#
#                             steps += 1
#                             if steps >= self.max_steps:
#                                 break
#
#                         return_list.append(ep_reward)
#                         cost_list.append(ep_cost)
#
#                     AVG_R.append(np.mean(return_list))
#                     AVG_C.append(np.mean(cost_list))
#
#
#                     best_metrics = {tb_name + 'Avg Return': np.mean(return_list), tb_name + 'Avg Cost': np.mean(cost_list)}
#                     wandb.log(best_metrics)
#
#             # Save model and save last trajectory
#             if (epoch % save_every == 0) or (epoch == self.clone_epochs - 1):
#                 logger.save_state({'env': env}, None)
#
#         xs = list([i for i in range(0, self.clone_epochs, eval_every)])
#         ys_expert_cost = list([1 * avg_expert_costs for _ in range(0, self.clone_epochs, eval_every)])
#         ys_expert_reward = list([1 * avg_expert_rewards for _ in range(0, self.clone_epochs, eval_every)])
#
#
#         ys_new = [AVG_R, ys_expert_reward]
#         zs_new = [AVG_C, ys_expert_cost]
#
#         rew_keys = ["Avg Clone Returns", "Avg Expert Returns"]
#         cost_keys = ["Avg Clone Costs", "Avg Expert Costs"]
#
#         rew_keys_mod = [ tb_name + sub for sub in rew_keys]
#         cost_keys_mod = [tb_name + sub for sub in cost_keys]
#
#         wandb.log({"rewards over training": line_series(xs=xs, ys=ys_new, keys=rew_keys_mod,
#                                                         title= tb_name + "Clone Rewards while Training")})
#         wandb.log(
#             {"costs over training": line_series(xs=xs, ys=zs_new, keys=cost_keys_mod,
#                                                 title= tb_name + "Clone Costs while Training")})
#
#         wandb.finish()
#
#     def run_clone_sim(self, env, record_clone, num_episodes, render, input_vector=[1,0]):
#         print(colorize("Running simulations of trained %s clone on %s environment over %d episodes" % (
#         self.config_name, env, num_episodes),
#                        'red', bold=True))
#
#         if record_clone:
#             # Logging
#             wandb.login()
#             wandb.init(memo=self.benchmark_memo_name, name=self.fname)
#
#             rew_mov_avg_10, cost_mov_avg_10, returns, costs = [], [], [], []
#
#             cum_ret, cum_cost = 0, 0
#
#             # Play clone episodes
#             for i in range(num_episodes):
#                 obs, done, totalr, totalc, steps = env.reset(), False, 0., 0., 0
#
#                 while not done:
#                     if self.expert_type == 'single':
#                         a = self.clone_policy(torch.tensor(obs).float())
#                     else:
#                         extend = np.array(input_vector)
#                         appended_obs = np.append(obs, extend, 0)
#
#                         a = self.clone_policy(torch.tensor(appended_obs).float())
#
#                     obs, r, done, info = env.step(a.detach().numpy())
#                     cost = info['cost']
#                     if render:
#                         env.render()
#                     totalr += r
#                     totalc += cost
#                     steps += 1
#                     if steps % 100 == 0: print("%i/%i" % (steps, self.max_steps))
#                     if steps >= self.max_steps:
#                         break
#                 returns.append(totalr)
#                 costs.append(totalc)
#
#                 cum_ret += totalr
#                 cum_cost += totalc
#
#                 if len(rew_mov_avg_10) >= 25:
#                     rew_mov_avg_10.pop(0)
#                     cost_mov_avg_10.pop(0)
#
#                 rew_mov_avg_10.append(totalr)
#                 cost_mov_avg_10.append(totalc)
#
#                 mov_avg_ret = np.mean(rew_mov_avg_10)
#                 mov_avg_cost = np.mean(cost_mov_avg_10)
#
#                 clone_metrics = {self.table_name + 'episode return': totalr,
#                                  self.table_name + 'episode cost': totalc,
#                                  self.table_name + '25ep mov avg return': mov_avg_ret,
#                                  self.table_name + '25ep mov avg cost': mov_avg_cost
#                                  }
#                 wandb.log(clone_metrics)
#
#             wandb.finish()
#
#             print('Returns', returns)
#             print('Avg EpRet', np.mean(returns))
#             print('Std EpRet', np.std(returns))
#             print('Costs', costs)
#             print('Avg EpCost', np.mean(costs))
#             print('Std EpCost', np.std(costs))
#
#
# ######################################################################################
#
#
# class DistillBehavioralClone(BehavioralClone):
#     def __init__(self, config_name_list, config_name,
#                  record_samples, clone_policy,
#                  optimizer, criterion,
#                  seed, expert_episodes,  clone_epochs, replay_buffer_size):
#
#         super().__init__(
#                  config_name,
#                  record_samples,
#                  clone_policy,
#                  optimizer,
#                  criterion,
#                  seed,
#                  expert_episodes,
#                  clone_epochs,
#                  replay_buffer_size)
#
#         self.config_name_list = config_name_list
#         self.n_experts = len(self.config_name_list)
#         self.benchmark_memo_name = 'distillppo_tests'
#         self.table_name = ''
#         self.expert_type = 'multiple'
#         # self.input_vector = input_vector
#
#     def set_multiple_replay_buffers(self, env):
#         print(self.config_name_list)
#
#         obs_dim = env.observation_space.shape
#         act_dim = env.action_space.shape
#
#         print(colorize("Pulling saved trajectories from two experts ( %s and %s) from files over %d episodes" %
#                        (self.config_name_list[0], self.config_name_list[1], self.expert_episodes), 'blue', bold=True))
#
#         rb_list = []
#
#         v = 0
#         for x in self.config_name_list:
#
#             _expert_demo_dir = os.path.join(self._expert_path, x + '_episodes/')
#
#             f = open(_expert_demo_dir + 'sim_data_' + str(self.expert_episodes) + '_buffer.pkl', "rb")
#             buffer_file = pickle.load(f)
#             f.close()
#
#             data = samples_from_cpprb(npsamples=buffer_file)
#
#             # Reconstruct the data, then pass it to replay buffer
#             np_states, np_rewards, np_actions, np_next_states, np_dones, np_next_dones = samples_to_np(data)
#
#             # Create environment
#             before_add = create_before_add_func(env)
#
#             replay_buffer = ReplayBuffer(size=self.replay_buffer_size,
#                                          env_dict={
#                                              "obs": {"shape": tuple([obs_dim[0]+2,])},
#                                              "act": {"shape": act_dim},
#                                              "rew": {},
#                                              "next_obs": {"shape": tuple([obs_dim[0]+2,])},
#                                              "done": {}})
#
#
#
#             # Concatenate the states with one hot vectors depending on class
#             extend1 = [one_hot(np.array([v]), self.n_experts)] * np_states[~np_dones].shape[0]
#
#             appended_states = np.append(np_states[~np_dones], np.c_[extend1], 1)
#             appended_next_states = np.append(np_next_states[~np_dones], np.c_[extend1], 1)
#
#             replay_buffer.add(**before_add(obs=appended_states,
#                                            act=np_actions[~np_dones],
#                                            rew=np_rewards[~np_dones],
#                                            next_obs=appended_next_states,
#                                            done=np_next_dones[~np_dones]))
#
#             rb_list.append(replay_buffer)
#             v += 1
#         self.rb_list = rb_list
#
#
#     def dualtrain_clone2(self, env, train_iters, batch_size, print_every=20, save_every=20, eval_sample_efficiency=False, eval_every=5, eval_episodes=20, exp_name= 'distilltest_rose_marigold_clone'):
#
#         self.fname = exp_name
#
#         for epoch in range(self.clone_epochs):
#             total_loss = 0
#
#             for t in range(train_iters):
#
#                 index = t % len(self.rb_list)
#
#                 SAMPLE = self.rb_list[index].sample(batch_size)
#
#                 states = SAMPLE['obs']
#                 actions = SAMPLE['act']
#
#                 self.optimizer.zero_grad()
#
#                 # Policy loss
#                 a_pred = self.clone_policy(torch.tensor(states).float())
#                 loss = self.criterion(a_pred, torch.tensor(actions))
#
#                 # print("Loss!", loss)
#                 total_loss += loss.item()
#                 loss.backward()
#                 if t % print_every == print_every - 1:
#                     print(colorize('Epoch:%d Batch:%d Loss:%.4f' % (epoch, t + 1, total_loss / print_every),
#                                    'yellow', bold=True))
#                     epoch_metrics = {'Avg Epoch Loss': total_loss / print_every}
#                     # wandb.log(epoch_metrics)
#                     total_loss = 0
#
#                 self.optimizer.step()
#
#
#
