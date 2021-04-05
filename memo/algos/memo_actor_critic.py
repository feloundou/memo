# Main entrance of MEMO
import time
import torch
import wandb
import gym
import safety_gym

import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

from memo.models.neural_nets import MEMO
from memo.utils.buffer_torch import MemoryBatch
from memo.utils.utils import proc_id, num_procs, EpochLogger, \
    setup_pytorch_for_mpi, sync_params, mpi_avg_grads, count_vars, \
    frange_cycle_sigmoid


####################################################3

def memo_valor(env_fn,
                model=MEMO,
                  memo_kwargs=dict(),
                  annealing_kwargs=dict(),
                  seed=0,
                  episodes_per_epoch=40,
                  epochs=50,
                  warmup=10,
                  train_iters=5,
                  memo_lr=1e-3,
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
                     str(memo_kwargs['encoder_hidden']) + 'DEC ' + str(memo_kwargs['decoder_hidden'])

    wandb.init(project="MEMO", group='Epochs: ' + str(epochs),  name=composite_name, config=locals())

    assert memories != [], "No examples found! Replay/memory buffers must be set to proceed."

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

    # Model    # Create discriminator and monitor it
    con_dim = len(memories)
    memo = model(obs_dim=obs_dim[0], out_dim=act_dim[0], **memo_kwargs)

    # Set up model saving
    logger.setup_pytorch_saver([memo])

    # Sync params across processes
    sync_params(memo)
    N_expert = episodes_per_epoch*max_ep_len
    print("N Expert: ", N_expert)

    # Buffer
    # local_episodes_per_epoch = int(episodes_per_epoch / num_procs())
    local_iter_per_epoch = int(train_iters / num_procs())

    # Count variables
    var_counts = tuple(count_vars(module) for module in [memo])
    logger.log('\nNumber of parameters: \t d: %d\n' % var_counts)

    # Optimizers
    memo_optimizer = Adam(memo.parameters(), lr=memo_lr)
    # scheduler = StepLR(memo_optimizer, step_size=2, gamma=0.96)
    scheduler = ReduceLROnPlateau(memo_optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    start_time = time.time()

    # Prepare data
    mem = MemoryBatch(memories)

    transition_states, pure_states, transition_actions, expert_ids = mem.collate()
    total_l_old, recon_l_old, context_l_old = 0, 0, 0

    # Main Loop
    kl_beta_schedule = frange_cycle_sigmoid(epochs, **annealing_kwargs)

    for epoch in range(epochs):
        memo.train()

        # Select state transitions and actions at random indexes
        batch_indexes = torch.randint(len(transition_states), (train_batch_size,))

        raw_states_batch, delta_states_batch, actions_batch, sampled_experts = \
           pure_states[batch_indexes], transition_states[batch_indexes], transition_actions[batch_indexes], expert_ids[batch_indexes]

        print("Expert IDs: ", sampled_experts)
        recon_gamma = 1e-8 if epoch < warmup else 1    # underweight recognition early on
        # recon_gamma = 1    # no warmup

        # for i in range(train_iters):
        for i in range(local_iter_per_epoch):
            # kl_beta = kl_beta_schedule[epoch]
            kl_beta = 1

            # only take context labeling into account for first label
            loss, recon_loss, X, latent_labels, vq_loss = memo(raw_states_batch, delta_states_batch,  actions_batch,
                                                                     kl_beta, recon_gamma)
            memo_optimizer.zero_grad()
            loss.mean().backward()
            mpi_avg_grads(memo)
            memo_optimizer.step()

        total_l_new, recon_l_new, vq_l_new = loss.mean().data.item(), recon_loss.mean().data.item(), vq_loss.mean().data.item()

        memo_metrics = {'MEMO Loss': total_l_new, 'Recon Loss': recon_l_new, "VQ Labeling Loss": vq_l_new,
                        "KL Beta": kl_beta_schedule[epoch],
                        # "LearningRate": lr[0]
                        }
        wandb.log(memo_metrics)

        logger.store(TotalLoss=total_l_new, PolicyLoss=recon_l_new, # ContextLoss=context_l_new,
                     DeltaTotalLoss=total_l_new-total_l_old, DeltaPolicyLoss=recon_l_new-recon_l_old,
                     )

        total_l_old, recon_l_old = total_l_new, recon_l_new  # , context_l_new

        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            logger.save_state({'env': env}, [memo], None)

        # Log
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpochBatchSize', train_batch_size)
        logger.log_tabular('TotalLoss', average_only=True)
        logger.log_tabular('PolicyLoss', average_only=True)
        logger.log_tabular('Time', time.time() - start_time)
        logger.dump_tabular()

    print("Finished training, and detected %d contexts!" % len(memo.found_contexts))
    # wandb.finish()
    return memo, mem



