import gym
import memo
from memo.models.neural_nets import MLPActorCritic
from memo.utils.agent_utils import Expert
from memo_valor import memo_valor
from memo.utils.utils import setup_logger_kwargs, mpi_fork
from memo.algos.demo_policies import spinning_top_policy, circle_policy, square_policy, \
    forward_policy, back_forth_policy, forward_spin_policy
from memo.utils.utils import memo_full_eval

# Run MEMO
ENV_NAME = 'Safexp-PointGoal1-v0'
# exp_name = "four-experts-memo-run-20" https://wandb.ai/openai-scholars/MEMO/runs/3fbo5yql?workspace=user-feloundou
# exp_name = "four-experts-memo-run-25" https://wandb.ai/openai-scholars/MEMO/runs/1efkebt9?workspace=user-feloundou
# exp_name = "four-experts-memo-backup-100" https://wandb.ai/openai-scholars/MEMO/runs/2tinwmwn?workspace=user-feloundou
# exp_name = "four-experts-memo-backup-400-prob" https://wandb.ai/openai-scholars/MEMO/runs/1xcuis1k?workspace=user-feloundou
# exp_name =  "two-experts-memo-50" https://wandb.ai/openai-scholars/MEMO/runs/k8nzpznu?workspace=user-feloundou
# exp_name = "two-experts-memo-100"  https://wandb.ai/openai-scholars/MEMO/runs/2iyxdoqh?workspace=user-feloundou
# exp_name = "two-experts-memo-5000" https://wandb.ai/openai-scholars/MEMO/runs/ejn3ge5p?workspace=user-feloundou
# exp_name = "two-experts-memo-10000"  https://wandb.ai/openai-scholars/MEMO/runs/22gffkb2?workspace=user-feloundou
# exp_name = "two-experts-memo-1000"
# exp_name = "2experts-2latents-memo-1000" https://wandb.ai/openai-scholars/MEMO/runs/2f8dvzko?workspace=user-feloundou
# exp_name = "2experts-3latents-memo-1000" https://wandb.ai/openai-scholars/MEMO/runs/24q65iel?workspace=user-feloundou
# exp_name = "2experts-4latents-memo-1000" https://wandb.ai/openai-scholars/MEMO/runs/2fxbfw61?workspace=user-feloundou
# exp_name = "2experts-5latents-memo-1000" https://wandb.ai/openai-scholars/MEMO/runs/2dvs9k0o?workspace=user-feloundou

exp_name = "memo-temp"

ep_len_config = 1000
cpu = 1
latent_modes_config=5
episodes_per_epoch_config=100
train_batch_size_config=500
eval_batch_size_config=100
seed_config=0
epochs_config=1000
warmup_config=100
train_iters_config=50
lr_config=3e-4
memo_kwargs_config=dict(encoder_hidden=[1000],
                        decoder_hidden=[64]*6,
                        actor_hidden=[256]*8,
                        latent_modes=latent_modes_config)

# 0. Make environment
env = gym.make(ENV_NAME)

# 1. Make Experts
greedy_expert = Expert(config_name='greedy', extension='-ppo',
                       record_samples=True, actor_critic=MLPActorCritic,
                       ac_kwargs=dict(hidden_sizes=[128]*4), seed=0)

forward_expert = Expert(config_name='forward', extension='-policy',
                        record_samples=True, actor_critic=MLPActorCritic,
                        ac_kwargs=dict(hidden_sizes=[128]*4), seed=0)

marigold_expert = Expert(config_name='marigold', extension='_128x4',
                record_samples=True, actor_critic=MLPActorCritic,
                ac_kwargs=dict(hidden_sizes=[128] * 4), seed=0) #444

# Seed here helps reset initialization for episodes for the perfect set of trajectories
rose_expert = Expert(config_name='rose', extension='_128x4',
                record_samples=True,  actor_critic=MLPActorCritic,
                ac_kwargs=dict(hidden_sizes=[128] * 4), seed=123)   # 123

circle_expert = Expert(config_name='circle', extension='',
                record_samples=True,  actor_critic=MLPActorCritic,
                ac_kwargs=dict(hidden_sizes=[128] * 4), seed=123)   # 123

# 2. Make the dataset (you can train them with expert.ppo_train() but these were pre-trained)
create_data = False
fetch_data = not create_data

greedy_expert.run_expert_sim(env=env, get_from_file=fetch_data,
                             episode_split=[10, 10],
                             expert_episodes=100, replay_buffer_size=10000,
                             seeds=[0, 444, 123, 999, 85, 4444, 64, 128, 808, 838])

forward_expert.run_expert_sim(env=env, get_from_file=fetch_data,
                             episode_split=[10, 10],
                             expert_episodes=100,  mode="demo", demo_pi=forward_policy,
                             seeds=[0, 444, 123, 999, 85, 4444, 64, 128, 808, 838])

# marigold_expert.run_expert_sim(env=env, get_from_file=fetch_data,
#                                episode_split=[10, 10],
#                                expert_episodes=100,  replay_buffer_size=10000,
#                                seeds=[0, 444, 123, 999, 85, 4444, 64, 128, 808, 838])
#
#
# # rose_expert.run_expert_sim(env=env, get_from_file=fetch_data,
# #                            episode_split=[10, 10],
# #                            expert_episodes=100, replay_buffer_size=10000,
# #                            seeds=[0, 444, 123, 999, 85, 4444, 64, 128, 808, 838])
#
# circle_expert.run_expert_sim(env=env, get_from_file=fetch_data,
#                              episode_split=[10, 10],
#                              expert_episodes=100, mode="demo", demo_pi=circle_policy,
#                              seeds=[0, 444, 123, 999, 85, 4444, 64, 128, 808, 838])
print("Replay Buffers Created")


# 3. Train
mpi_fork(cpu)
logger_kwargs = setup_logger_kwargs(exp_name, 0)

#
memo, c_memories = memo_valor(lambda: gym.make(ENV_NAME),
           seed=seed_config,
           memo_kwargs=memo_kwargs_config,
           annealing_kwargs=dict(start=0., stop=1., n_cycle=1, ratio=0.5),
           episodes_per_epoch=episodes_per_epoch_config,   # fix reward accumulation
           max_ep_len=ep_len_config,
           epochs=epochs_config,
           warmup=warmup_config,
           train_iters=train_iters_config,
           memo_lr=lr_config,  # ideal
           train_batch_size=train_batch_size_config,
           eval_batch_size=eval_batch_size_config,
           logger_kwargs=logger_kwargs,
           memories=[greedy_expert.memory, forward_expert.memory])




# # # Good run: exp_name: memo-marigold-rose: https://wandb.ai/openai-scholars/MEMO/runs/h8755blk?workspace=user-feloundou
# memo, c_memories = memo_valor(lambda: gym.make(ENV_NAME),
#            seed=seed_config,
#            memo_kwargs=memo_kwargs_config,
#            annealing_kwargs=dict(start=0., stop=1., n_cycle=10, ratio=0.5),
#            # episodes_per_epoch=100,   # fix reward accumulation
#            episodes_per_epoch=episodes_per_epoch_config,   # fix reward accumulation
#            max_ep_len=ep_len_config,
#            epochs=epochs_config,
#            warmup=warmup_config,
#            train_iters=train_iters_config,
#            memo_lr=lr_config,  # ideal
#            # memo_lr=3e-5,  # ideal 3e-5 works well
#            # memo_lr=5e-5,  # Learning rate appears to be the most important hyperparameter right now. The smaller the better.
#            # memo_lr=3e-4,   # Karpathy Konstant
#            train_batch_size=train_batch_size_config,
#            eval_batch_size=eval_batch_size_config,
#            logger_kwargs=logger_kwargs,
#            memories=[greedy_expert.memory, forward_expert.memory,
#                      marigold_expert.memory, circle_expert.memory])

# , rose_expert.memory,


# Write some assertion code to check determinism in each episode.
# -> Fixed number of experts, what happens if
# the number of experts is greater than or less than number of modes?

# Promising run:
# 1. https://wandb.ai/openai-scholars/MEMO/runs/2xl9srlb?workspace=user-feloundou (with annealing)
# 2. https://wandb.ai/openai-scholars/MEMO/runs/3ga8ux7g?workspace=user-feloundou (same as 1 but without annealing)
# 4. Evaluate
# memo_full_eval(model=memo, expert_names=['greedy', 'forward', 'marigold', 'rose', 'circle'],
#                file_names=[greedy_expert.file_name, forward_expert.file_name,
#                            marigold_expert.file_name, rose_expert.file_name,
#                            circle_expert.file_name],
#                pi_types=['policy', 'demo', 'policy', 'policy', 'demo'],
#                collated_memories=c_memories, latent_modes=latent_modes_config,
#                # eval_modes=['class', 'policy', 'quantitative'],
#                eval_modes=['class', 'policy'],
#                episodes_per_epoch=episodes_per_epoch_config, quant_episodes=10,
#                N_expert=episodes_per_epoch_config*ep_len_config,
#                eval_batch_size=eval_batch_size_config, seed=seed_config,
#                logger_kwargs=logger_kwargs)

# memo_full_eval(model=memo, expert_names=['greedy', 'forward', 'marigold', 'circle'],
#                file_names=[greedy_expert.file_name, forward_expert.file_name,
#                            marigold_expert.file_name, circle_expert.file_name],
#                pi_types=['policy', 'demo', 'policy', 'demo'],
#                collated_memories=c_memories, latent_modes=latent_modes_config,
#                # eval_modes=['class', 'policy', 'quantitative'],
#                eval_modes=['class', 'policy'],
#                # eval_modes=['policy'],
#                episodes_per_epoch=episodes_per_epoch_config, quant_episodes=10,
#                N_expert=episodes_per_epoch_config*ep_len_config,
#                eval_batch_size=eval_batch_size_config, seed=seed_config,
#                logger_kwargs=logger_kwargs)

memo_full_eval(model=memo, expert_names=['greedy', 'forward'],
               file_names=[greedy_expert.file_name, forward_expert.file_name],
               pi_types=['policy', 'demo'],
               collated_memories=c_memories, latent_modes=latent_modes_config,
               # eval_modes=['class', 'policy', 'quantitative'],
               eval_modes=['class', 'policy'],
               # eval_modes=['policy'],
               episodes_per_epoch=episodes_per_epoch_config, quant_episodes=10,
               N_expert=episodes_per_epoch_config*ep_len_config,
               eval_batch_size=eval_batch_size_config, seed=seed_config,
               logger_kwargs=logger_kwargs)

# Some tips to train MEMO:
# 1. # Warmup improves performance
# 2. # Smaller training rates are better
# 3. # Generate as much data as your computer can handle.


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

# This repo relies on PyTorch and has some dependencies that must be pulled in separately.



# larger batches <- better results:
# compare 2000 vs 5000 samples:
# 2000: https://wandb.ai/openai-scholars/MEMO/runs/1clgi4lv?workspace=user-feloundou
# 5000:  https://wandb.ai/openai-scholars/MEMO/runs/2sq4a1qw?workspace=user-feloundou

# number of iterations (with the 256 network) improves performance:
# 100  https://wandb.ai/openai-scholars/MEMO/runs/2uaconf9?workspace=user-feloundou
# 200  https://wandb.ai/openai-scholars/MEMO/runs/1c7w8yom?workspace=user-feloundou
# 200 with more data: https://wandb.ai/openai-scholars/MEMO/runs/3rrjava4
# 200 with shallower actor network:


# the larger the batches, the more iterations you probably need

# network size:
# 500: https://wandb.ai/openai-scholars/MEMO/runs/1clgi4lv?workspace=user-feloundou
# 2048: https://wandb.ai/openai-scholars/MEMO/runs/1j2iwi86?workspace=user-feloundou
