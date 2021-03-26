from abc import ABC, abstractmethod

# from cpprb import ReplayBuffer, create_before_add_func
import wandb
import os.path as osp
from memo.utils.buffer_torch import Trajectory, Buffer

import numpy as np
import torch
import os, time, pickle
# import time
#
from memo.models.neural_nets import MLPActorCritic
from torch.optim import Adam
from memo.utils.torch_cpo_utils import ExpertSinglePathSimulator
#
# # from ppo_algos import MLPActorCritic
#
from memo.utils.utils import setup_logger_kwargs, setup_pytorch_for_mpi, \
    mpi_avg, mpi_avg_grads, mpi_sum, num_procs, colorize, EpochLogger, \
    CostPOBuffer, proc_id, load_policy_and_env, run_policy
#
# from run_policy_sim_ppo import load_policy_and_env, run_policy
#

# Set up function for computing PPO policy loss
def compute_loss_pi_utils(data, ac, clip_ratio):
    obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

    # Policy loss
    pi, logp = ac.pi(obs, act)
    ratio = torch.exp(logp - logp_old)
    clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
    loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

    # Useful extra info
    approx_kl = (logp_old - logp).mean().item()
    ent = pi.entropy().mean().item()
    clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
    clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
    pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

    return loss_pi, pi_info

# Set up functions for computing value loss(es)
def compute_loss_v_utils(data, ac):
    obs, ret, cret = data['obs'], data['ret'], data['cret']
    v_loss = ((ac.v(obs) - ret) ** 2).mean()
    return v_loss


def update_ppo(ac, cur_penalty, clip_ratio, logger, buf, train_pi_iters, pi_optimizer,
                                         train_v_iters, vf_optimizer, cost_lim, penalty_lr,
                                         rew_mov_avg, cost_mov_avg):
    cur_cost = logger.get_stats('EpCost')[0]
    cur_rew = logger.get_stats('EpRet')[0]

    if len(rew_mov_avg) >= 10:
        rew_mov_avg.pop(0)
        cost_mov_avg.pop(0)

    rew_mov_avg.append(cur_rew)
    cost_mov_avg.append(cur_cost)

    mov_avg_ret = np.mean(rew_mov_avg)
    mov_avg_cost = np.mean(cost_mov_avg)

    c = cur_cost - cost_lim
    # c is the safety constraint
    print("current cost: ", cur_cost)

    data = buf.get()

    pi_l_old, pi_info_old = compute_loss_pi_utils(data, ac, clip_ratio)
    pi_l_old = pi_l_old.item()
    v_l_old = compute_loss_v_utils(data, ac).item()

    # Train policy with multiple steps of gradient descent
    for i in range(train_pi_iters):
        pi_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi_utils(data, ac, clip_ratio)
        # kl = mpi_avg(pi_info['kl'])
        loss_pi.backward()
        mpi_avg_grads(ac.pi)  # average grads across MPI processes
        pi_optimizer.step()

    logger.store(StopIter=i)

    # Value function learning
    for i in range(train_v_iters):
        vf_optimizer.zero_grad()
        loss_v = compute_loss_v_utils(data, ac)
        loss_v.backward()
        mpi_avg_grads(ac.v)  # average grads across MPI processes
        vf_optimizer.step()

    # Penalty update
    cur_penalty = max(0, cur_penalty + penalty_lr * (cur_cost - cost_lim))

    # Log changes from update
    kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
    logger.store(LossPi=pi_l_old, LossV=v_l_old,
                 KL=kl, Entropy=ent, ClipFrac=cf,
                 DeltaLossPi=(loss_pi.item() - pi_l_old),
                 DeltaLossV=(loss_v.item() - v_l_old))

    vf_loss_avg = mpi_avg(v_l_old)
    pi_loss_avg = mpi_avg(pi_l_old)

    return cur_penalty, mov_avg_ret, mov_avg_cost, vf_loss_avg, pi_loss_avg


class Agent(ABC):
    """
    Abstract clone class
    """

    @abstractmethod
    def ppo_train(self):
        pass

    @abstractmethod
    def run_expert_sim(self):
        """Run episodes from a pre-trained clone policy"""
        pass


class Expert(Agent):
    """
    Clone class for Sampler.
    """

    def __init__(self,
                 config_name,
                 record_samples,
                 actor_critic=MLPActorCritic,
                 ac_kwargs=dict(),
                 seed=0,
                 penalty_init=5e-3,
                 ):

        self.config_name = config_name
        self.record_samples = record_samples
        self.seed = seed
        self.replay_buffer = None
        self.actor_critic = actor_critic
        self.ac_kwargs = ac_kwargs

        self.penalty_init = penalty_init
        self.penalty_init_param = np.log(max(np.exp(self.penalty_init) - 1, 1e-8))

        self.max_steps = 1000

        # Paths

        self._memo_dir = osp.abspath(osp.dirname(osp.dirname(__file__)))
        self._root_data_path = osp.join(self._memo_dir, 'data')
        self._expert_path = osp.join(self._memo_dir, 'expert_data')
        self._clone_path = osp.join(self._memo_dir, 'clone_data')
        self._demo_dir = osp.join(self._expert_path, self.config_name + '_episodes/')
        self.file_name = 'ppo_penalized_' + self.config_name + '_128x4'

        # Special function to avoid certain slowdowns from PyTorch + MPI combo.
        setup_pytorch_for_mpi()

        # Random seed # seed = 0
        self.seed += 10000 * proc_id()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

    def ppo_train(self, env_fn, epochs, gamma, lam,
    steps_per_epoch, train_pi_iters, pi_lr, train_vf_iters, vf_lr, penalty_lr, cost_lim, clip_ratio,
                  max_ep_len, save_every, wandb_write=True, logger_kwargs=dict()):

        # 4 million env interactions
        if wandb_write:
            wandb.init(memo="TrainingExperts", group="ppo_runs", name='ppo_pen_' + self.config_name)

        # Set up logger
        logger = EpochLogger(**logger_kwargs)
        logger.save_config(locals())

        # Make environment
        env = env_fn()

        obs_dim = env.observation_space.shape
        act_dim = env.action_space.shape

        self.ac = self.actor_critic(env.observation_space, env.action_space, **self.ac_kwargs)

        # Set up Torch saver for logger setup
        logger.setup_pytorch_saver(self.ac)

        if wandb_write:
            wandb.watch(self.ac)

        # Set up experience buffer
        self.local_steps_per_epoch = int(steps_per_epoch / num_procs())
        self.buf = CostPOBuffer(obs_dim, act_dim, self.local_steps_per_epoch, gamma, lam)

        # Set up optimizers for policy and value function
        pi_optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr)
        vf_optimizer = Adam(self.ac.v.parameters(), lr=vf_lr)

        penalty = np.log(max(np.exp(self.penalty_init) - 1, 1e-8))

        mov_avg_ret, mov_avg_cost = 0, 0

        # Prepare for interaction with environment
        start_time = time.time()
        o, r, d, c, ep_ret, ep_cost, ep_len, cum_cost, cum_reward = env.reset(), 0, False, 0, 0, 0, 0, 0, 0
        rew_mov_avg, cost_mov_avg = [], []

        cur_penalty = self.penalty_init_param

        for epoch in range(epochs):
            for t in range(self.local_steps_per_epoch):
                a, v, vc, logp = self.ac.step(torch.as_tensor(o, dtype=torch.float32))


                # env.step => Take action
                next_o, r, d, info = env.step(a)

                # Include penalty on cost
                c = info.get('cost', 0)

                # Track cumulative cost over training
                cum_reward += r
                cum_cost += c

                ep_ret += r
                ep_cost += c
                ep_len += 1

                r_total = r - cur_penalty * c
                r_total /= (1 + cur_penalty)

                self.buf.store(o, a, r_total, v, 0, 0, logp, info)

                # save and log
                logger.store(VVals=v)

                # Update obs (critical!)
                o = next_o

                timeout = ep_len == max_ep_len
                terminal = d or timeout
                epoch_ended = t == self.local_steps_per_epoch - 1

                if terminal or epoch_ended:
                    if epoch_ended and not terminal:
                        print('Warning: trajectory cut off by epoch at %d steps.' % ep_len, flush=True)
                    # if trajectory didn't reach terminal state, bootstrap value target
                    if timeout or epoch_ended:
                        _, v, _, _ = self.ac.step(torch.as_tensor(o, dtype=torch.float32))
                        last_v = v
                        last_vc = 0

                    else:
                        last_v = 0

                    self.buf.finish_path(last_v, last_vc)

                    if terminal:
                        # only save EpRet / EpLen if trajectory finished
                        print("end of episode return: ", ep_ret)
                        logger.store(EpRet=ep_ret, EpLen=ep_len, EpCost=ep_cost)

                        # average ep ret and cost
                        avg_ep_ret = ep_ret
                        avg_ep_cost = ep_cost
                        episode_metrics = {'average ep ret': avg_ep_ret, 'average ep cost': avg_ep_cost}

                        if wandb_write:
                            wandb.log(episode_metrics)

                    # Reset environment
                    o, r, d, c, ep_ret, ep_len, ep_cost = env.reset(), 0, False, 0, 0, 0, 0

            # Save model and save last trajectory
            if (epoch % save_every == 0) or (epoch == epochs - 1):
                logger.save_state({'env': env}, None)

            # Perform PPO update!
            cur_penalty, mov_avg_ret, mov_avg_cost, vf_loss_avg, pi_loss_avg = \
                                         update_ppo(self.ac, cur_penalty, clip_ratio,
                                                    logger,
                                                    self.buf,
                                                    train_pi_iters, pi_optimizer,
                                                    train_vf_iters, vf_optimizer,
                                                    cost_lim, penalty_lr,
                                                    rew_mov_avg, cost_mov_avg)

            if wandb_write:
                update_metrics = {'10p mov avg ret': mov_avg_ret,
                                  '10p mov avg cost': mov_avg_cost,
                                  'value function loss': vf_loss_avg,
                                  'policy loss': pi_loss_avg,
                                  'current penalty': cur_penalty
                                  }

                wandb.log(update_metrics)
            #  Cumulative cost calculations
            cumulative_cost = mpi_sum(cum_cost)
            cumulative_reward = mpi_sum(cum_reward)

            cost_rate = cumulative_cost / ((epoch + 1) * steps_per_epoch)
            reward_rate = cumulative_reward / ((epoch + 1) * steps_per_epoch)

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('EpCost', with_min_and_max=True)
            logger.log_tabular('VVals', with_min_and_max=True)
            logger.log_tabular('TotalEnvInteracts', (epoch + 1) * steps_per_epoch)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossV', average_only=True)
            logger.log_tabular('DeltaLossPi', average_only=True)
            logger.log_tabular('DeltaLossV', average_only=True)
            logger.log_tabular('Entropy', average_only=True)
            logger.log_tabular('KL', average_only=True)
            logger.log_tabular('ClipFrac', average_only=True)
            logger.log_tabular('StopIter', average_only=True)
            logger.log_tabular('Time', time.time() - start_time)
            logger.dump_tabular()

            if wandb_write:
                log_metrics = {'cost rate': cost_rate, 'reward rate': reward_rate}
                wandb.log(log_metrics)

                wandb.finish()


    def run_expert_sim(self, env, get_from_file, max_cost=None, min_reward=None, episode_split=[1,1], expert_episodes=1000, replay_buffer_size=10000, seeds=[0]):
        """Episode split [number of episodes per iteration, number of iterations]"""
        obs_dim = env.observation_space.shape
        act_dim = env.action_space.shape

        print("Now adding trajectories to memory")

        # n_trajectories = expert_episodes
        n_trajectories = episode_split[0]
        trajectory_len = 1000

        if get_from_file:
            print(colorize("Pulling saved expert %s trajectories from file over %d episodes" %
                           (self.config_name, expert_episodes), 'blue', bold=True))

            m_file = osp.join(self._expert_path , self.config_name + '_episodes/memory_data_' + str(
                int(expert_episodes)) + '_buffer.pkl')

            file_m = open(m_file, 'rb')
            memory = pickle.load(file_m)
            self.memory = memory
            file_m.close()

        else:
            print("N trajectories! ", n_trajectories)
            # Generate expert data
            print(colorize(
                "Generating expert %s trajectories from file over %d episodes" % (
                self.config_name, expert_episodes), 'blue', bold=True))

            # Load trained policy
            _, expert_pi = load_policy_and_env(osp.join(self._root_data_path, self.file_name, self.file_name + '_s0/'),
                                                'last', False)

            print("Hello, Simulator for !", self.config_name)

            self.simulator = ExpertSinglePathSimulator('Safexp-PointGoal1-v0',
                                                       expert_pi, n_trajectories, trajectory_len)

            print("running the simulation and saving to 'memory' ")
            # all_memories = []
            AggTraj = None
            n_iter = episode_split[1]

            for i in range(n_iter):
                self.memory, traj = self.simulator.run_sim(sampling_mode=False, seed=seeds[i])
                AggTraj = traj if AggTraj is None else np.append(AggTraj, traj)

            BigMemory = Buffer(AggTraj)
            bufname_pk = self._expert_path + self.config_name + '_episodes/memory_data_' + str(int(n_trajectories*n_iter)) + '_buffer.pkl'
            file_pi = open(bufname_pk, 'wb')
            pickle.dump(BigMemory, file_pi)

            # self.memory, traj = self.simulator.run_sim(sampling_mode=False, seed=0)

            # all_rewards, all_costs = [], []
            #
            # for k in range(n_trajectories):
            #     all_rewards.append(torch.stack(self.memory[k].rewards, dim=0).sum(dim=0).item())
            #     all_costs.append(torch.stack(self.memory[k].costs, dim=0).sum(dim=0).item())
            #
            # print("All memory rewards: ",  [round(num, 2) for num in all_rewards] )
            # print("All memory costs: ", [round(num, 2) for num in all_costs])

            # Save to pickle
            # bufname_pk = self._expert_path + self.config_name + '_episodes/memory_data_' + str(int(n_trajectories)) + '_buffer.pkl'
            # file_pi = open(bufname_pk, 'wb')
            # pickle.dump(self.memory, file_pi)

            # # second
            # self.memory2, traj2 = self.simulator.run_sim(sampling_mode=False, seed=444)
            # all_rewards, all_costs = [], []
            # for k in range(n_trajectories):
            #     all_rewards.append(torch.stack(self.memory2[k].rewards, dim=0).sum(dim=0).item())
            #     all_costs.append(torch.stack(self.memory2[k].costs, dim=0).sum(dim=0).item())
            #
            # print("All memory rewards: ", [round(num, 2) for num in all_rewards])
            # print("All memory costs: ", [round(num, 2) for num in all_costs])

            # # Save to pickle
            # bufname_pk = self._expert_path + self.config_name + '_episodes/memory2_data_' + str(
            #     int(n_trajectories)) + '_buffer.pkl'
            # file_pi = open(bufname_pk, 'wb')
            # pickle.dump(self.memory, file_pi)



#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
