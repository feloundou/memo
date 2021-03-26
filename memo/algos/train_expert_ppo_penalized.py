# Main entrance of GAIL
import numpy as np
import torch
import torch.nn.functional as F
from adabelief_pytorch import AdaBelief
import gym
import safety_gym
import time


from memo.models.neural_nets import ActorCritic, count_vars

from memo.utils.utils import BufferActor
from memo.utils.utils import mpi_fork, proc_id, num_procs, EpochLogger,\
    average_gradients, sync_all_params, setup_pytorch_for_mpi, sync_params, mpi_avg_grads
    # compute_loss_policy

import wandb

def ppo_penalized(env_fn,
            actor_critic=ActorCritic,
            ac_kwargs=dict(),
            seed=0,
            episodes_per_epoch=40,
            epochs=500,
            gamma=0.99,
            lam=0.98,
            pi_lr=3e-4,
            vf_lr=1e-3,
            train_v_iters=80,
            train_pi_iters=1,  ## NOTE: Incredibly Important That This Be Low For Penalized Learning
            max_ep_len=1000,
            logger_kwargs=dict(),
            clip_ratio = 0.2, # tuned????
        # Cost constraints / penalties:
                cost_lim=25,
                penalty_init=1.,
                penalty_lr=5e-3,
            config_name='standard',
            save_freq=10):

    # W&B Logging
    wandb.login()

    composite_name = 'new_ppo_penalized_' + config_name
    wandb.init(memo="LearningCurves", group="PPO Expert", name=composite_name)

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    ac_kwargs['action_space'] = env.action_space

    # Models
    # Create actor-critic module and monitor it
    ac = actor_critic(input_dim=obs_dim[0], **ac_kwargs)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    # Sync params across processes
    sync_params(ac)

    # Buffers
    local_episodes_per_epoch = int(episodes_per_epoch / num_procs())
    buf = BufferActor(obs_dim[0], act_dim[0], local_episodes_per_epoch, max_ep_len)

    # Count variables
    var_counts = tuple(count_vars(module) for module in [ac.pi, ac.v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)

    # Optimizers
    pi_optimizer = torch.optim.Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = torch.optim.Adam(ac.v.parameters(), lr=vf_lr)

    # pi_optimizer = AdaBelief(ac.pi.parameters(), betas=(0.9, 0.999), eps=1e-8)
    # vf_optimizer = AdaBelief(ac.v.parameters(), betas=(0.9, 0.999), eps=1e-8)


    # # Parameters Sync
    # sync_all_params(ac.parameters())

    # Set up function for computing PPO policy loss
    def compute_loss_pi(obs, act, adv, logp_old):
        # Policy loss # policy gradient term + entropy term
        # Policy loss with clipping (without clipping, loss_pi = -(logp*adv).mean()).
        # TODO: Think about removing clipping
        _, logp, _ = ac.pi(obs, act)

        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        return loss_pi


    def penalty_update(cur_penalty):
        cur_cost = logger.get_stats('EpCost')[0]
        cur_rew = logger.get_stats('EpRet')[0]

        # Penalty update
        cur_penalty = max(0, cur_penalty + penalty_lr * (cur_cost - cost_lim))
        return cur_penalty

    def update(e):
        obs, act, adv, ret, logp_old = [torch.Tensor(x) for x in buf.retrieve_all()]

        # Policy
        _, logp, _ = ac.pi(obs, act)
        entropy = (-logp).mean()

        # Train policy with multiple steps of gradient descent
        for _ in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi = compute_loss_pi(obs, act, adv, ret)
            loss_pi.backward()
            # average_gradients(train_pi.param_groups)
            # mpi_avg_grads(pi_optimizer.param_groups)
            mpi_avg_grads(ac.pi)
            pi_optimizer.step()

        # Value function training
        v = ac.v(obs)
        v_l_old = F.mse_loss(v, ret)  # old loss

        for _ in range(train_v_iters):
            v = ac.v(obs)
            v_loss = F.mse_loss(v, ret)   # how well did our value function predict loss?

            # Value function train
            vf_optimizer.zero_grad()
            v_loss.backward()
            # average_gradients(vf_optimizer.param_groups)
            mpi_avg_grads(ac.v)   # average gradients across MPI processes
            vf_optimizer.step()

        # Log the changes
        _, logp, _, v = ac(obs, act)
        # entropy_new = (-logp).mean()
        pi_loss_new = -(logp * adv).mean()
        v_loss_new = F.mse_loss(v, ret)
        kl = (logp_old - logp).mean()
        logger.store(LossPi=loss_pi, LossV=v_l_old, DeltaLossPi=(pi_loss_new - loss_pi),
                     DeltaLossV=(v_loss_new - v_l_old), Entropy=entropy, KL=kl)

    # Prepare for interaction with the environment
    start_time = time.time()
    o, r, d, ep_ret, ep_cost, ep_len = env.reset(), 0, False, 0, 0, 0
    total_t = 0

    # Initialize penalty
    cur_penalty = np.log(max(np.exp(penalty_init) - 1, 1e-8))

    for epoch in range(epochs):
        ac.eval()     # eval mode
        # Policy rollout
        for _ in range(local_episodes_per_epoch):
            for _ in range(max_ep_len):

                # obs =
                a, _, lopg_t, v_t = ac(torch.Tensor(o.reshape(1, -1)))

                logger.store(VVals=v_t)

                o, r, d, info = env.step(a.detach().numpy()[0])

                c = info.get('cost', 0)  # Include penalty on cost

                r_total = r - cur_penalty * c
                r_total /= (1 + cur_penalty)

                # store
                buf.store(o, a.detach().numpy(), r_total, v_t.item(), lopg_t.detach().numpy())

                ep_ret += r
                ep_cost += c
                ep_len += 1
                total_t += 1

                terminal = d or (ep_len == max_ep_len)
                if terminal:
                    # buf.end_episode()
                    buf.finish_path()
                    logger.store(EpRet=ep_ret, EpCost=ep_cost, EpLen=ep_len)

                    print("end of episode return: ", ep_ret)



                    episode_metrics = {'average ep ret': ep_ret, 'average ep cost': ep_cost}
                    wandb.log(episode_metrics)

                    o, r, d, ep_ret, ep_cost, ep_len = env.reset(), 0, False, 0, 0, 0

        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            # logger._torch_save(ac, fname="expert_torch_save.pt")
            # logger._torch_save(ac, fname="model.pt")
            logger.save_state({'env': env}, None, None)

        # Update
        ac.train()

        # update penalty
        cur_penalty = penalty_update(cur_penalty)

        # update networks
        update(epoch)

        # Log
        logger.log_tabular('Epoch', epoch)
        # logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpRet', average_only=True)
        logger.log_tabular('EpCost', average_only=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', average_only=True)
        # logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', total_t)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        # logger.log_tabular('KL', average_only=True)
        logger.log_tabular('Time', time.time() - start_time)
        logger.dump_tabular()

    wandb.finish()





if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Safexp-PointGoal1-v0')
    parser.add_argument('--hid', type=int, default=128)
    parser.add_argument('--l', type=int, default=4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lam', type=float, default=0.95)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=2)
    parser.add_argument('--episodes-per-epoch', type=int, default=20)
    parser.add_argument('--cost-lim', type=int, default=25)
    # parser.add_argument('--episodes-per-epoch', type=int, default=40)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--exp_name', type=str, default='test-pen-ppo')
    args = parser.parse_args()

    mpi_fork(args.cpu)

    from memo.utils.utils import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    ppo_penalized(lambda: gym.make(args.env), actor_critic=ActorCritic, ac_kwargs=dict(hidden_dims=[args.hid]*args.l),
        gamma=args.gamma, lam=args.lam, seed=args.seed, episodes_per_epoch=args.episodes_per_epoch,
        epochs=args.epochs, logger_kwargs=logger_kwargs, cost_lim = args.cost_lim)
