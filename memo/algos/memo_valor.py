# Main entrance of GAIL
import numpy as np
import gym
from PIL import Image
from numpy import asarray
import safety_gym
import pandas as pd
import time, random, torch, wandb
from torch.distributions import Independent, OneHotCategorical, Categorical
import torch.nn.functional as F
import wandb.plot as wplot
from torch.optim import Adam, SGD, lr_scheduler
import wandb
import os
from plotnine import ggplot, aes, geom_path, geom_smooth
# import traj_dist
import traj_dist.distance as tdist
from memo.utils.utils import run_memo_eval, run_memo_quant

# (Source: https://github.com/bguillouet/traj-dist/blob/master/traj_dist/)
from memo.models.neural_nets import MEMO
from memo.utils.buffer_torch import MemoryBatch

from memo.utils.utils import mpi_fork, proc_id, num_procs, EpochLogger, \
    setup_pytorch_for_mpi, sync_params, mpi_avg_grads, count_vars,  \
    frange_cycle_sigmoid


####################################################3

def memo_valor(env_fn,
                vae=MEMO,
                  vaelor_kwargs=dict(),
                  annealing_kwargs=dict(),
                  seed=0,
                  episodes_per_epoch=40,
                  epochs=50,
                  train_valor_iters=5,
                  vae_lr=1e-3,
                  train_batch_size=50,
                  eval_batch_size=200,
                  max_ep_len=1000,
                  logger_kwargs=dict(),
                  config_name='standard',
                  save_freq=10,
               # replay_buffers=[],
               memories=[]):
    # W&B Logging
    wandb.login()

    composite_name = 'E ' + str(epochs) + ' B ' + str(train_batch_size) + ' ENC ' + \
                     str(vaelor_kwargs['encoder_hidden']) + 'DEC ' + str(vaelor_kwargs['decoder_hidden'])

    wandb.init(project="MEMO", group='Epochs: ' + str(epochs),  name=composite_name, config=locals())

    assert memories != [], "Replay/memory buffers must be set"

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
    # con_dim = len(replay_buffers)
    con_dim = len(memories)
    # con_dim = 10   # con_dim = 1
    memo = vae(obs_dim=obs_dim[0], latent_dim=con_dim, out_dim=act_dim[0], **vaelor_kwargs)

    # Set up model saving
    logger.setup_pytorch_saver([memo])

    # Sync params across processes
    sync_params(memo)
    N_expert = episodes_per_epoch*max_ep_len
    print("N Expert: ", N_expert)

    # Buffer
    # local_episodes_per_epoch = int(episodes_per_epoch / num_procs())
    local_iter_per_epoch = int(train_valor_iters / num_procs())

    # Count variables
    var_counts = tuple(count_vars(module) for module in [memo])
    logger.log('\nNumber of parameters: \t d: %d\n' % var_counts)

    # Optimizers
    vae_optimizer = Adam(memo.parameters(), lr=vae_lr)
    pi_optimizer = 0

    start_time = time.time()

    # Prepare data
    mem = MemoryBatch(memories)
    # print("Length of one buffer: ", memories[0].length)

    transition_states, pure_states, transition_actions, expert_ids = mem.collate()
    valor_l_old, recon_l_old, context_l_old = 0, 0, 0

    # context_dist = OneHotCategorical(logits=torch.Tensor(np.ones(2)))
    context_dist = Categorical(logits=torch.Tensor(np.ones(con_dim)))

    # Main Loop
    kl_beta_schedule = frange_cycle_sigmoid(epochs, **annealing_kwargs)   # kl_beta_schedule = frange_cycle_linear(epochs, n_cycle=10)

    for epoch in range(epochs):
        memo.train()

        o_tensor = context_dist.sample((train_batch_size,))
        o_onehot = F.one_hot(o_tensor, con_dim).squeeze().float()

        # Select state transitions and actions at random indexes
        batch_indexes = torch.randint(len(transition_states), (train_batch_size,))

        raw_states_batch, delta_states_batch, actions_batch, sampled_experts = \
           pure_states[batch_indexes], transition_states[batch_indexes], transition_actions[batch_indexes], expert_ids[batch_indexes]

        print("Expert IDs: ", sampled_experts)
        # print("Actual actions: ", actions_batch)

        warmup = 1000
        recon_gamma = 1e-8 if epoch < warmup else 1    #underweight recognition early on

        # for i in range(train_valor_iters):
        for i in range(local_iter_per_epoch):
        #     kl_beta = 0 if i > 0 else kl_beta_schedule[epoch]
            kl_beta = kl_beta_schedule[epoch]
            # only take context labeling into account for first label

            loss, recon_loss, X, latent_labels, vq_valor_loss = memo(raw_states_batch, delta_states_batch,  actions_batch, o_onehot, con_dim,
                                                                   kl_beta, recon_gamma)
            vae_optimizer.zero_grad()
            loss.mean().backward()
            mpi_avg_grads(memo)
            vae_optimizer.step()

        valor_l_new, recon_l_new, vq_l_new = loss.mean().data.item(), recon_loss.mean().data.item(), vq_valor_loss.mean().data.item()

        vaelor_metrics = {'VALOR Loss': valor_l_new, 'Recon Loss': recon_l_new, "VQ Labeling Loss": vq_l_new, "KL Beta": kl_beta_schedule[epoch]}
        wandb.log(vaelor_metrics)

        logger.store(VALORLoss=valor_l_new, PolicyLoss=recon_l_new, # ContextLoss=context_l_new,
                     DeltaValorLoss=valor_l_new-valor_l_old, DeltaPolicyLoss=recon_l_new-recon_l_old,
                     # DeltaContextLoss=context_l_new-context_l_old
                     )

        # logger.store(VALORLoss = d_loss)
        valor_l_old, recon_l_old = valor_l_new, recon_l_new  # , context_l_new

        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            logger.save_state({'env': env}, [memo], None)

        # Log
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpochBatchSize', train_batch_size)
        logger.log_tabular('VALORLoss', average_only=True)
        logger.log_tabular('PolicyLoss', average_only=True)
        logger.log_tabular('Time', time.time() - start_time)
        logger.dump_tabular()

#########
    # Run eval
    print("RUNNING Classification EVAL")
    memo.eval()

    fake_o = context_dist.sample((eval_batch_size*len(memories),))  #eval batch size for each expert type
    fake_o_onehot = F.one_hot(fake_o, con_dim).squeeze().float()

    # Randomize and fetch an evaluation sample
    eval_raw_states_batch, eval_delta_states_batch, eval_actions_batch, eval_sampled_experts = \
         mem.eval_batch(N_expert, eval_batch_size, episodes_per_epoch)

    # Pass through VAELOR
    loss, recon_loss, _, predicted_expert_labels, _ = memo(eval_raw_states_batch,
                                                             eval_delta_states_batch,
                                                             eval_actions_batch,
                                                             fake_o_onehot, con_dim)

    ground_truth, predictions = eval_sampled_experts, predicted_expert_labels

    print("ground truth", np.array(ground_truth))
    print("predictions ", np.array(predictions))

    # Confusion matrix
    class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    wandb.log({"confusion_mat": wplot.confusion_matrix(
        y_true=np.array(ground_truth), preds=np.array(predictions), class_names=class_names)})


# ###############
    print("RUNNING POLICY EVAL")

    # unroll and plot a full episode
    marigold_path_image, rose_path_image, marigold_actions_image, rose_actions_image, \
    images_traj, images_actions, marigold_traj_data, rose_traj_data, marigold_action_data, rose_action_data =\
        run_memo_eval(exp_name=logger_kwargs['exp_name'], experts=['marigold', 'rose'],
                     contexts=list(range(10)), seed=7, env_fn=lambda: gym.make('Safexp-PointGoal1-v0'))

    marigold_dist_table = wandb.Table(data=marigold_traj_data, columns=["label", "value"])
    rose_dist_table = wandb.Table(data=rose_traj_data, columns=["label", "value"])

    wandb.log({"Expert paths": [wandb.Image(asarray(marigold_path_image), caption="Marigold path"),
                                wandb.Image(asarray(rose_path_image), caption="Rose path")],
               "Learners paths": [wandb.Image(asarray(img), caption="Learners path") for img in images_traj],
               "Expert actions": [wandb.Image(asarray(marigold_actions_image), caption="Marigold actions"),
                                  wandb.Image(asarray(rose_actions_image), caption="Rose actions")],
               "Learners actions": [wandb.Image(asarray(img), caption="Learners actions") for img in images_actions],
               "Marigold Dist": wandb.plot.bar(marigold_dist_table, "label", "value", title="Marigold Distances"),
               "Rose Dist": wandb.plot.bar(rose_dist_table, "label", "value", title="Rose Distances")})


# #####
    print("Running Quantitative Eval")
    run_memo_quant(exp_name=logger_kwargs['exp_name'], experts=['marigold', 'rose'],
                  contexts=list(range(10)), num_episodes=100, seed=7, env_fn=lambda: gym.make('Safexp-PointGoal1-v0'))


    wandb.finish()



#
# if __name__ == '__main__':
#     import argparse
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--env', type=str, default='Safexp-PointGoal1-v0')
#     parser.add_argument('--hid', type=int, default=128)
#     parser.add_argument('--l', type=int, default=2)
#     parser.add_argument('--seed', '-s', type=int, default=0)
#     parser.add_argument('--cpu', type=int, default=1)
#     parser.add_argument('--episodes-per-epoch', type=int, default=5)
#     parser.add_argument('--epochs', type=int, default=1000)
#     parser.add_argument('--exp_name', type=str, default='valor-anonymous-expert')
#     args = parser.parse_args()
#
#     mpi_fork(args.cpu)
#
#     from memo.utils.utils import setup_logger_kwargs
#
#     logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
#
#     memo_valor(lambda: gym.make(args.env),
#                   seed=args.seed, episodes_per_epoch=args.episodes_per_epoch,
#                   epochs=args.epochs,
#                   logger_kwargs=logger_kwargs)
