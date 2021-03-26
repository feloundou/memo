# Main entrance of GAIL
import numpy as np
import torch
import torch.nn.functional as F
import gym
import safety_gym
import time
import os.path as osp

from torch.distributions.categorical import Categorical

from memo.models.neural_nets import ActorCritic, ValorDiscriminator

import wandb

from memo.utils.utils import VALORBuffer
from memo.utils.utils import mpi_fork, proc_id, num_procs, EpochLogger,\
     setup_pytorch_for_mpi, sync_params, mpi_avg_grads, count_vars,  mpi_sum

def valor_penalized(env_fn, actor_critic=ActorCritic, ac_kwargs=dict(),
          disc=ValorDiscriminator, dc_kwargs=dict(), seed=0,
          episodes_per_epoch=40,
          epochs=50, gamma=0.99, pi_lr=3e-4, vf_lr=1e-3, dc_lr=5e-4,
          train_pi_iters=1, train_v_iters=80, train_dc_iters=10,
          train_dc_interv=10,
          lam=0.97,
          # Cost constraints / penalties:
          cost_lim=50,
          penalty_init=1.,
          penalty_lr=5e-3,
          clip_ratio=0.2,
          max_ep_len=1000, logger_kwargs=dict(), con_dim=5, config_name='standard',
          save_freq=10, k=1):
    # W&B Logging
    wandb.login()

    composite_name = 'new_valor_penalized_' + config_name
    wandb.init(memo="LearningCurves", group="VALOR Expert", name=composite_name)

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

    # Model    # Create actor-critic modules and discriminator and monitor them
    ac = actor_critic(input_dim=obs_dim[0] + con_dim, **ac_kwargs)
    discrim = disc(input_dim=obs_dim[0], context_dim=con_dim, **dc_kwargs)

    # Set up model saving
    logger.setup_pytorch_saver([ac, discrim])

    # Sync params across processes
    sync_params(ac)
    sync_params(discrim)

    # Buffer
    local_episodes_per_epoch = int(episodes_per_epoch / num_procs())
    buffer = VALORBuffer(con_dim, obs_dim[0], act_dim[0], local_episodes_per_epoch, max_ep_len, train_dc_interv)

    # Count variables
    var_counts = tuple(count_vars(module) for module in  [ac.pi, ac.v, discrim.pi])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d, \t d: %d\n' % var_counts)

    # Optimizers
    pi_optimizer = torch.optim.Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = torch.optim.Adam(ac.v.parameters(), lr=vf_lr)
    discrim_optimizer = torch.optim.Adam(discrim.pi.parameters(), lr=dc_lr)

    def compute_loss_pi(obs, act, adv, logp_old):
        # Policy loss # policy gradient term + entropy term
        # Policy loss with clipping (without clipping, loss_pi = -(logp*adv).mean()).
        # TODO: Think about removing clipping
        _, logp, _ = ac.pi(obs, act)

        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        return loss_pi

    # # Parameters Sync
    # sync_all_params(ac.parameters())
    # sync_all_params(disc.parameters())

    def penalty_update(cur_penalty):  # update penalty
        cur_cost = logger.get_stats('EpCost')[0]
        cur_penalty = max(0, cur_penalty + penalty_lr * (cur_cost - cost_lim))
        return cur_penalty

    def update(e):
        obs, act, adv, pos, ret, cost, logp_old = [torch.Tensor(x) for x in buffer.retrieve_all()]

        # cur_cost = logger.get_stats('EpCost')[0]
        # cur_rew = logger.get_stats('EpRet')[0]

        mov_avg_ret = ret.sum(axis=-1)/(episodes_per_epoch)
        mov_avg_cost = cost.sum(axis=-1)/(episodes_per_epoch)

        # Policy
        print("policy pull")
        # _, logp, _ = ac.pi(obs, act)
        # entropy = (-logp).mean()

        # Train policy with multiple steps of gradient descent
        for _ in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi = compute_loss_pi(obs, act, adv, ret)
            loss_pi.backward()
            mpi_avg_grads(ac.pi)
            pi_optimizer.step()


        # Value function
        print("value function pull")
        v = ac.v(obs)
        v_l_old = F.mse_loss(v, ret)

        for _ in range(train_v_iters):
            v = ac.v(obs)
            v_loss = F.mse_loss(v, ret)

            # Value function train
            vf_optimizer.zero_grad()
            v_loss.backward()
            mpi_avg_grads(ac.v)
            vf_optimizer.step()

        # Discriminator
        if (e + 1) % train_dc_interv == 0:
            print('Discriminator Update!')
            # Remove BiLSTM, take FFNN, take state_diff and predict what the context was
            # Predict what was the context based on the tuple (or just context from just the current state)

            con, s_diff = [torch.Tensor(x) for x in buffer.retrieve_dc_buff()]
            print("s diff: ", s_diff)
            print("s diff shape: ", s_diff.shape)
            _, logp_dc, _ = discrim(s_diff, con)
            d_l_old = -logp_dc.mean()

            # Discriminator train
            for _ in range(train_dc_iters):
                _, logp_dc, _ = discrim(s_diff, con)
                d_loss = -logp_dc.mean()
                discrim_optimizer.zero_grad()
                d_loss.backward()
                mpi_avg_grads(discrim.pi)
                discrim_optimizer.step()

            _, logp_dc, _ = discrim(s_diff, con)
            dc_l_new = -logp_dc.mean()
        else:
            d_l_old = 0
            dc_l_new = 0

        # Log the changes
        # print("logging changes")
        _, logp, _, v = ac(obs, act)
        # _, logp, _ = ac.pi(obs, act)
        pi_l_new = -(logp * (k * adv + pos)).mean()
        v_l_new = F.mse_loss(v, ret)
        kl = (logp_old - logp).mean()
        logger.store(LossPi=loss_pi, LossV=v_l_old, KL=kl,
                     # Entropy=entropy,
                     DeltaLossPi=(pi_l_new - loss_pi),
                     DeltaLossV=(v_l_new - v_l_old),
                     LossDC=d_l_old,
                     DeltaLossDC=(dc_l_new - d_l_old))

        update_metrics = {'10p mov avg ret': mov_avg_ret,
                          '10p mov avg cost': mov_avg_cost,
                          'current penalty': cur_penalty
                          }

        wandb.log(update_metrics)

        # logger.store(Adv=adv.reshape(-1).numpy().tolist(), Pos=pos.reshape(-1).numpy().tolist())

    start_time = time.time()
    o, r, d, ep_ret, ep_cost, ep_len, cum_reward, cum_cost = env.reset(), 0, False, 0, 0, 0, 0, 0
    rew_mov_avg, cost_mov_avg = [], []
    context_dist = Categorical(logits=torch.Tensor(np.ones(con_dim)))
    print("context distribution:", context_dist)
    total_t = 0

    # Initialize penalty
    cur_penalty = np.log(max(np.exp(penalty_init) - 1, 1e-8))

    for epoch in range(epochs):
        ac.eval()
        discrim.eval()
        for _ in range(local_episodes_per_epoch):
            c = context_dist.sample()
            print("context sample: ", c)
            c_onehot = F.one_hot(c, con_dim).squeeze().float()
            # print("one hot sample: ", c_onehot)

            for _ in range(max_ep_len):
                concat_obs = torch.cat([torch.Tensor(o.reshape(1, -1)), c_onehot.reshape(1, -1)], 1)

                a, _, logp_t, v_t = ac(concat_obs)

                o, r, d, info = env.step(a.detach().numpy()[0])
                # print("info", info)
                # time.sleep(0.002)
                cost = info.get("cost")
                if cost is None:
                    print("Hey! Cost NoneType Error: ", info)
                    print(info)
                    print("What was the reward? :", r)
                    # print("What was the action? :", a.detach())

                ep_cost += cost
                ep_ret += r
                ep_len += 1
                total_t += 1

                cum_reward += r
                cum_cost += cost

                r_total = r - cur_penalty * cost
                r_total /= (1 + cur_penalty)

                buffer.store(c, concat_obs.squeeze().detach().numpy(), a.detach().numpy(), r_total, cost, v_t.item(),
                             logp_t.detach().numpy())

                # buffer.store(c, concat_obs.squeeze().detach().numpy(), a.detach().numpy(), r, v_t.item(),
                #              logp_t.detach().numpy())
                logger.store(VVals=v_t)

                terminal = d or (ep_len == max_ep_len)
                if terminal:
                    dc_diff = torch.Tensor(buffer.calc_diff()).unsqueeze(0)
                    con = torch.Tensor([float(c)]).unsqueeze(0)
                    # _, _, log_p = discrim(dc_diff, con)
                    print("context going into discrim:", con)
                    _, log_p, _ = discrim(dc_diff, con)
                    # look at the bug, should take the second output instead of log_p ///
                    # instead of doing average over sequence dimension, do not need to average reward,
                    # just give reward now
                    # with pure VAE, do not have to calculate advantages

                    buffer.finish_path(log_p.detach().numpy())
                    logger.store(EpRet=ep_ret, EpCost=ep_cost, EpLen=ep_len)

                    episode_metrics = {'average ep ret': ep_ret, 'average ep cost': ep_cost}

                    wandb.log(episode_metrics)

                    o, r, d, ep_ret, ep_cost, ep_len = env.reset(), 0, False, 0, 0, 0

        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            logger.save_state({'env': env}, [ac, discrim], None)

        # Update
        ac.train()
        discrim.train()

        # update penalty
        cur_penalty = penalty_update(cur_penalty)

        # update models
        update(epoch)

        #  Cumulative cost calculations
        cumulative_cost = mpi_sum(cum_cost)
        cumulative_reward = mpi_sum(cum_reward)

        cost_rate = cumulative_cost / ((epoch + 1) * episodes_per_epoch * max_ep_len)
        reward_rate = cumulative_reward / ((epoch + 1) * episodes_per_epoch * max_ep_len)

        log_metrics = {'cost rate': cost_rate, 'reward rate': reward_rate}
        wandb.log(log_metrics)

        # Log
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpCost', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', total_t)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('LossDC', average_only=True)
        logger.log_tabular('DeltaLossDC', average_only=True)
        # logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('Time', time.time() - start_time)
        logger.dump_tabular()

    wandb.finish()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Safexp-PointGoal1-v0')
    parser.add_argument('--hid', type=int, default=128)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lam', type=float, default=0.97)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--episodes-per-epoch', type=int, default=5)
    # parser.add_argument('--episodes-per-epoch', type=int, default=40)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--exp_name', type=str, default='valor-anonymous-expert')
    parser.add_argument('--con', type=int, default=5)
    args = parser.parse_args()

    mpi_fork(args.cpu)

    from memo.utils.utils import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    valor_penalized(lambda: gym.make(args.env), actor_critic=ActorCritic,
                    ac_kwargs=dict(hidden_dims=[args.hid] * args.l),
          disc=ValorDiscriminator, dc_kwargs=dict(hidden_dims=[args.hid]*args.l),
                    # dc_kwargs=dict(hidden_dims=[args.hid]),
          gamma=args.gamma, seed=args.seed, episodes_per_epoch=args.episodes_per_epoch, epochs=args.epochs,
          logger_kwargs=logger_kwargs, con_dim=args.con)
