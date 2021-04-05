import sys
import torch
from torch.optim import Adam
import numpy as np
import time
from adabelief_pytorch import AdaBelief

import gym
import safety_gym
# from safety_gym.envs.engine import Engine

from memo.models.neural_nets import MLPActorCritic
from memo.utils.utils import EpochLogger, setup_pytorch_for_mpi, sync_params, mpi_avg_grads, \
    mpi_fork, mpi_avg, mpi_sum,  num_procs, proc_id, count_vars, CostPOBuffer
# from memo.models.ppo_algos import *
# from agent_types import *
from memo.algos.demo_policies import spinning_top_policy, circle_policy, square_policy, \
    forward_policy, back_forth_policy, forward_spin_policy

# Define PPO functions
def ppo(env_fn,
        actor_critic=MLPActorCritic,
        demo_pi=square_policy,
        # agent=PPOAgent(),
        ac_kwargs=dict(),
        seed=0,
        # Experience Collection
        steps_per_epoch=4000,
        epochs=50,
        max_ep_len=1000,
        # Discount factors:
        gamma=0.99,
        lam=0.97,
        cost_gamma=0.99,
        cost_lam=0.97,
        # Policy Learning:
        ent_reg=0.,
        # Cost constraints / penalties:
        cost_lim=25,
        penalty_init=1.,
        penalty_lr=5e-3,
        # KL divergence:
        target_kl=0.01,
        # Value learning:
        vf_lr=1e-3,
        train_v_iters=100,
        # Policy Learning:
        pi_lr=3e-4,
        train_pi_iters=100,
        # Clipping
        clip_ratio=0.2,
        logger_kwargs=dict(),
        # Experimenting
        config_name = 'standard',
        save_every=10):
    """
    """

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Create actor-critic module and monitor it
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)

    # Sync params across processes
    sync_params(ac)

    # Count variables
    var_counts = tuple(count_vars(module) for module in [ac.pi, ac.v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = CostPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam, cost_gamma, cost_lam)

    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    mov_avg_ret, mov_avg_cost = 0, 0

    # Set up function for computing PPO policy loss
    def compute_loss_pi(data):
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
    def compute_loss_v(data):
        obs, ret, cret = data['obs'], data['ret'], data['cret']
        v_loss = ((ac.v(obs) - ret) ** 2).mean()
        return v_loss

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update():
        cur_cost = logger.get_stats('EpCost')[0]
        cur_rew = logger.get_stats('EpRet')[0]

        if len(rew_mov_avg_10) >= 10:
            rew_mov_avg_10.pop(0)
            cost_mov_avg_10.pop(0)

        rew_mov_avg_10.append(cur_rew)
        cost_mov_avg_10.append(cur_cost)

        c = cur_cost - cost_lim
        print("current cost: ", cur_cost)

        data = buf.get()

        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            kl = mpi_avg(pi_info['kl'])
            loss_pi.backward()
            mpi_avg_grads(ac.pi)  # average grads across MPI processes
            pi_optimizer.step()

        logger.store(StopIter=i)

        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            mpi_avg_grads(ac.v)  # average grads across MPI processes
            vf_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old))

        vf_loss_avg = mpi_avg(v_l_old)
        pi_loss_avg = mpi_avg(pi_l_old)

    # Prepare for interaction with environment
    start_time = time.time()
    o, r, d, c, ep_ret, ep_cost, ep_len, cum_cost, cum_reward = env.reset(), 0, False, 0, 0, 0, 0, 0, 0
    rew_mov_avg_10 = []
    cost_mov_avg_10 = []

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        step = 0

        for t in range(local_steps_per_epoch):
            # a, v, vc, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))
            a, v, vc, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))
            # print("action taken: ", a)
            # env.step => Take action
            next_o, r, d, info = env.step(a)
            # if epoch % 2:
            #     a = demo_pi[step]
            #     next_o, _, d, info = env.step(a)
            #     r = 0.5
            # else:
            #     next_o, _, d, info = env.step(a)
            #     r = -0.5

            # Include penalty on cost
            c = info.get('cost', 0)

            # Track cumulative cost over training
            cum_reward += r
            cum_cost += c

            ep_ret += r
            ep_cost += c
            ep_len += 1
            step += 1

            buf.store(o, a, r, v, 0, 0, logp, info)

            # save and log
            logger.store(VVals=v)

            # Update obs (critical!)
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t == local_steps_per_epoch - 1

            if terminal or epoch_ended:
                if epoch_ended and not terminal:
                    print('Warning: trajectory cut off by epoch at %d steps.' % ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, v, _, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                    last_v = v
                    last_vc = 0

                else:
                    last_v = 0
                buf.finish_path(last_v, last_vc)

                if terminal:
                    print("end of episode return: ", ep_ret)
                    logger.store(EpRet=ep_ret, EpLen=ep_len, EpCost=ep_cost)

                    # average ep ret and cost
                    avg_ep_ret = ep_ret
                    avg_ep_cost = ep_cost

                # Reset environment
                o, r, d, c, ep_ret, ep_len, ep_cost = env.reset(), 0, False, 0, 0, 0, 0
                step = 0

        # Save model and save last trajectory
        if (epoch % save_every == 0) or (epoch == epochs - 1):
            logger.save_state({'env': env}, None)

        # Perform PPO update!
        update()

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


def main(config):
    import argparse
    from memo.utils.utils import setup_logger_kwargs

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Safexp-PointGoal1-v0')
    parser.add_argument('--agent', type=str, default='ppo-lagrange')
    parser.add_argument('--hid', type=int, default=128)
    parser.add_argument('--l', type=int, default=4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--cost_gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    # parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--steps', type=int, default=8000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--cost_lim', type=float, default=25)
    parser.add_argument('--config_name', type=str, default='standard')
    parser.add_argument('--penalty_lr', type=float, default=0.005)
    parser.add_argument('--exp_name', type=str, default='greedy-ppo')

    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi
    # Set the demo policy
    # demo_pi = square_policy
    # logger_kwargs = setup_logger_kwargs("square_policy", args.seed)
    demo_pi = back_forth_policy
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)


    # Run experiment
    ppo(lambda: gym.make(args.env),
        actor_critic=MLPActorCritic,
        demo_pi=demo_pi,
        ac_kwargs=dict(hidden_sizes=[config['hid']] * config['l']),
        gamma=config['gamma'],
        lam=config['lam'],
        cost_gamma=args.cost_gamma,
        seed=config['seed'],
        steps_per_epoch=config['steps'],
        epochs=args.epochs,
        cost_lim= config['cost_lim'],
        penalty_lr=config['penalty_lr'],
        config_name= config['name'],
        logger_kwargs=logger_kwargs)

if __name__ == '__main__':
    demo_config = dict(name='demo', penalty_lr=0.01, cost_lim=0,
                       gamma=0.99, lam=0.95, seed=0, steps=20000,
                       hid=128, l=8)
    main(demo_config)

