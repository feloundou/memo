import numpy as np
import gym
import safety_gym
import time, random, torch, wandb
from torch.distributions import Independent, OneHotCategorical, Categorical
import torch.nn.functional as F
import wandb.plot as wplot
from torch.optim import Adam, SGD, lr_scheduler

from memo.models.neural_nets import VAELOR
from memo.utils.utils import mpi_fork, proc_id, num_procs, EpochLogger, \
    setup_pytorch_for_mpi, sync_params, mpi_avg_grads, count_vars, MemoryBatch


####################################################3

def vanilla_valor(env_fn,
                  vae=VAELOR,
                  # disc = ValorDiscriminator,
                  dc_kwargs=dict(),
                  seed=0,
                  episodes_per_epoch=40,
                  epochs=50,
                  # vae_lr=3e-4,
                  vae_lr = 1e-3,
                  train_batch_size = 50,
                  eval_batch_size=200,
                  train_valor_iters=200,
                  max_ep_len=1000,
                  logger_kwargs=dict(),
                  config_name='standard',
                  save_freq=10, replay_buffers=[], memories=[]):
    # W&B Logging
    wandb.login()

    composite_name = str(epochs) + 'epochs_' + str(train_batch_size) + 'batch_' + str(train_valor_iters) + 'iters'
    wandb.init(memo="Vanilla Valor", group='Epochs: ' + str(epochs), name=composite_name)

    assert replay_buffers != [], "Replay buffers must be set"

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

    # name = input("What is your name? ")
    # if (name != ""):
    #     # Print welcome message if the value is not empty
    #     print("Hello %s, welcome to playing with VAELOR" % name)

    # Model    # Create discriminator and monitor it
    con_dim = len(replay_buffers)
    # valor_vae = vae(obs_dim=env.observation_space.shape[0], latent_dim=con_dim, act_dim=2)
    valor_vae = vae(obs_dim=env.observation_space.shape[0], latent_dim=con_dim)

    con_dim = len(replay_buffers)
    # vae_discrim = disc(input_dim=obs_dim[0], context_dim=con_dim, **dc_kwargs)

    # Set up model saving
    logger.setup_pytorch_saver([valor_vae])

    # Sync params across processes
    sync_params(valor_vae)

    N_expert = episodes_per_epoch*1000

    # Buffer
    local_episodes_per_epoch = int(episodes_per_epoch / num_procs())

    # Count variables
    var_counts = tuple(count_vars(module) for module in [valor_vae])
    logger.log('\nNumber of parameters: \t d: %d\n' % var_counts)

    # Optimizers
    vae_optimizer = Adam(valor_vae.parameters(), lr=vae_lr)
    # steps = train_valor_iters
    # scheduler = lr_scheduler.CosineAnnealingLR(vae_optimizer, steps)

    context_optimizer = Adam(valor_vae.lmbd.parameters(), lr=vae_lr)
    action_optimizer = Adam(valor_vae.decoder.parameters(), lr=vae_lr)

    start_time = time.time()

    # Prepare data
    mem = MemoryBatch(memories)
    transition_states, pure_states, transition_actions, expert_ids = mem.collate()

    valor_l_old, recon_l_old, context_l_old = 0, 0, 0

    # context_dist = OneHotCategorical(logits=torch.Tensor(np.ones(2)))
    context_dist = Categorical(logits=torch.Tensor(np.ones(2)))

# Main Loop
    for epoch in range(epochs):
        valor_vae.train()
        ##

        c = context_dist.sample()  # this causes context learning to collapse very quickly
        c_onehot = F.one_hot(c, con_dim).squeeze().float()

        o_tensor = context_dist.sample_n(train_batch_size)
        o_onehot = F.one_hot(o_tensor, con_dim).squeeze().float()

        # Select state transitions and actions at random indexes
        batch_indexes = torch.randint(len(transition_states), (train_batch_size,))
        raw_states_batch, delta_states_batch, actions_batch, sampled_experts = \
           pure_states[batch_indexes], transition_states[batch_indexes], transition_actions[batch_indexes], expert_ids[batch_indexes]

        print("Expert IDs: ", sampled_experts[:2])
        # Train the VAE encoder and decoder

        min_loss = 1
        iter=0

        for _ in range(train_valor_iters):  # original

            vae_optimizer.zero_grad()
            # loss, recon_loss, context_loss, _, _ = valor_vae.compute_latent_loss(raw_states_batch, delta_states_batch,
            #                                                                      actions_batch, c_onehot)

            loss, recon_loss, context_loss, _, _ = valor_vae.compute_latent_loss(raw_states_batch, delta_states_batch,
                                                                                 actions_batch, o_onehot)

            # loss.backward(torch.ones_like(loss))
            # loss.sum().backward()
            loss.mean().backward()

            # loss_no_mean = loss   ## instead of meaning the loss
            # for i in range(loss.shape[-1]):
            #     vae_optimizer.zero_grad()
            #     loss_no_mean[i].backward(retain_graph=True)

            vae_optimizer.step()
            min_loss = context_loss.mean().data.item()
            iter += 1
            print("Epoch: \t", epoch, "\t Iter: \t", iter, "\t Context Loss: \t", min_loss, end='\n')

            # scheduler.step()
            # print("Scheduler LR: ", scheduler.get_lr())

        # print('Reset scheduler')
        # scheduler = lr_scheduler.CosineAnnealingLR(vae_optimizer, steps)

        # valor_l_new, recon_l_new, context_l_new = loss.sum().data.item(), recon_loss.sum().data.item(), context_loss.sum().data.item()
        valor_l_new, recon_l_new, context_l_new = loss.mean().data.item(), recon_loss.mean().data.item(), context_loss.mean().data.item()

        vaelor_metrics = {'VALOR Loss': valor_l_new, 'Recon Loss': recon_l_new, 'Context Loss': context_l_new}
        wandb.log(vaelor_metrics)

        logger.store(VALORLoss=valor_l_new, PolicyLoss=recon_l_new, ContextLoss=context_l_new,
                     DeltaValorLoss=valor_l_new-valor_l_old, DeltaPolicyLoss=recon_l_new-recon_l_old,
                     DeltaContextLoss=context_l_new-context_l_old
                     )

        # logger.store(VALORLoss = d_loss)
        valor_l_old, recon_l_old, context_l_old = valor_l_new, recon_l_new, context_l_new

        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            logger.save_state({'env': env}, [valor_vae], None)

        # # Log
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpochBatchSize', train_batch_size)
        logger.log_tabular('VALORLoss', average_only=True)
        logger.log_tabular('PolicyLoss', average_only=True)
        logger.log_tabular('ContextLoss', average_only=True)
        # logger.log_tabular('DeltaValorLoss', average_only=True)
        # logger.log_tabular('DeltaPolicyLoss', average_only=True)
        # logger.log_tabular('DeltaContextLoss', average_only=True)
        logger.log_tabular('Time', time.time() - start_time)
        logger.dump_tabular()



#########
    # Run eval
    print("RUNNING Classification EVAL")
    print("Total episodes per expert: ", N_expert)
    valor_vae.eval()
    fake_c = context_dist.sample()
    fake_c_onehot = F.one_hot(fake_c, con_dim).squeeze().float()

    # Randomize and fetch an evaluation sample
    eval_raw_states_batch, eval_delta_states_batch, eval_actions_batch, eval_sampled_experts = \
         mem.eval_batch(N_expert, eval_batch_size, episodes_per_epoch)

    # Pass through VAELOR
    loss, recon_loss, kl_loss, _, latent_v = valor_vae.compute_latent_loss(eval_raw_states_batch, eval_delta_states_batch,
                                                                                 eval_actions_batch, fake_c_onehot)
    # print("Latent V: ", latent_v)
    predicted_expert_labels = np.argmax(latent_v, axis=1)  # convert from one-hot
    ground_truth, predictions = eval_sampled_experts, predicted_expert_labels
    print("ground truth", ground_truth)
    print("predictions ", predictions)

    # Confusion matrix
    class_names = ["0", "1"]
    wandb.log({"confusion_mat": wplot.confusion_matrix(
        y_true=np.array(ground_truth), preds=np.array(predictions), class_names=class_names)})

    print("RUNNING POLICY EVAL")  # unroll and plot a full episode
    # (for now, selecting first episode in first memory)
    # pick some arbitary  episode starting point. Where does the episode start, and follow the episode for

    eval_observations, eval_actions, _, _ = memories[0].sample()
    one_ep_states, one_ep_actions = eval_observations[:1000], eval_actions[:1000]
    x_actions, y_actions = map(list, zip(*one_ep_actions))
    ep_time = torch.arange(1000)

    # Plot expert steps and actions
    x_expert_data = [[x, y] for (x, y) in zip(ep_time, x_actions)]
    y_expert_data = [[x, y] for (x, y) in zip(ep_time, y_actions)]

    x_table, y_table = wandb.Table(data=x_expert_data, columns=["x", "y"]), wandb.Table(data=y_expert_data, columns=["x", "y"]),

    # Collect learner experiences, give the network a state observation and a fixed label tensor
    learner_actions0, learner_actions1, tensor_tag0, tensor_tag1 = [], [], F.one_hot(torch.as_tensor(0), con_dim).float(), \
                                                F.one_hot(torch.as_tensor(1), con_dim).float()  # TODO: vary tensor_tag

    for step in range(1000):
        action_dist0 = valor_vae.decoder(torch.cat([one_ep_states[step], tensor_tag0], dim=-1))
        action_dist1 = valor_vae.decoder(torch.cat([one_ep_states[step], tensor_tag1], dim=-1))

        sampled_action0, sampled_action1 = action_dist0.sample(), action_dist1.sample()
        learner_actions0.append(sampled_action0)
        learner_actions1.append(sampled_action1)

    learner_x_actions0, learner_y_actions0 = map(list, zip(*learner_actions0))
    learner_x_actions1, learner_y_actions1 = map(list, zip(*learner_actions1))

    x_learner_data0 = [[x, y] for (x, y) in zip(ep_time, learner_x_actions0)]
    y_learner_data0 = [[x, y] for (x, y) in zip(ep_time, learner_y_actions0)]

    x_learner_data1 = [[x, y] for (x, y) in zip(ep_time, learner_x_actions1)]
    y_learner_data1 = [[x, y] for (x, y) in zip(ep_time, learner_y_actions1)]

    learner_x_table0, learner_y_table0 = wandb.Table(data=x_learner_data0, columns=["x", "y"]), \
                                       wandb.Table(data=y_learner_data0, columns=["x", "y"])

    learner_x_table1, learner_y_table1 = wandb.Table(data=x_learner_data1, columns=["x", "y"]), \
                                         wandb.Table(data=y_learner_data1, columns=["x", "y"])


    wandb.log({"Tracing Dimension 1": wandb.plot.scatter(x_table, "x", "y",
                                                       title="Expert Behavior over Time (X plane)"),
               "Tracing Dimension 2": wandb.plot.scatter(y_table, "x", "y",
                                                          title="Expert Behavior over Time (Y plane)"),
               "Learner0 Tracing Dimension 1": wandb.plot.scatter(learner_x_table0, "x", "y",
                                                         title="Student [0] Behavior over Time (X plane)"),
               "Learner0 Tracing Dimension 2": wandb.plot.scatter(learner_y_table0, "x", "y",
                                                         title="Student [0] Behavior over Time (Y plane)"),
               "Learner1 Tracing Dimension 1": wandb.plot.scatter(learner_x_table1, "x", "y",
                                                                 title="Student [1] Behavior over Time (X plane)"),
               "Learner1 Tracing Dimension 2": wandb.plot.scatter(learner_y_table1, "x", "y",
                                                                 title="Student [1] Behavior over Time (Y plane)")
               })

    wandb.finish()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Safexp-PointGoal1-v0')
    parser.add_argument('--hid', type=int, default=128)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--episodes-per-epoch', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--exp_name', type=str, default='valor-anonymous-expert')
    args = parser.parse_args()

    mpi_fork(args.cpu)

    from memo.utils.utils import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    vanilla_valor(lambda: gym.make(args.env),
                  dc_kwargs=dict(hidden_dims=[args.hid] * args.l),
                  seed=args.seed, episodes_per_epoch=args.episodes_per_epoch,
                  epochs=args.epochs,
                  logger_kwargs=logger_kwargs)