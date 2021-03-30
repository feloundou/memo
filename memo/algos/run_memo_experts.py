import gym
import memo
from neural_nets import MLPActorCritic
from memo.utils.agent_utils import Expert
from memo_valor import memo_valor
from memo.utils.utils import setup_logger_kwargs, mpi_fork
from memo.algos.demo_policies import spinning_top_policy, circle_policy, square_policy, \
    forward_policy, back_forth_policy, forward_spin_policy
from memo.utils.utils import memo_full_eval

# Run MEMO
ENV_NAME = 'Safexp-PointGoal1-v0'
exp_name = "vm-memo-run-200"
ep_len_config = 1000
cpu = 1
episodes_per_epoch_config=100
eval_batch_size_config=100
seed_config=0
latent_modes_config=5
memo_kwargs_config=dict(encoder_hidden=[1000],
                              decoder_hidden=[512],
                              latent_modes=latent_modes_config)
train_batch_size_config=20



# 0. Make environment
env = gym.make(ENV_NAME)

# 1. Make Experts
marigold_expert = Expert(config_name='marigold', extension='_128x4',
                record_samples=True, actor_critic=MLPActorCritic,
                ac_kwargs=dict(hidden_sizes=[128] * 4), seed=0, ) #444

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

if fetch_data:
    # marigold
    marigold_expert.run_expert_sim(env=env, get_from_file=True, expert_episodes=100)

    # rose
    rose_expert.run_expert_sim(env=env, get_from_file=True, expert_episodes=100)

    # circle
    circle_expert.run_expert_sim(env=env, get_from_file=True, expert_episodes=100)
    print("Replay Buffers Fetched")

else:
    # Run simulation to collect demonstrations (just a gut-check here)
    marigold_expert.run_expert_sim(env=env, get_from_file=False,
                                   episode_split=[10, 10],
                                   expert_episodes=100,  replay_buffer_size=10000,
                                   seeds=[0, 444, 123, 999, 85, 4444, 64, 128, 808, 838])

    rose_expert.run_expert_sim(env=env, get_from_file=False,
                               episode_split=[10, 10],
                               expert_episodes=100, replay_buffer_size=10000,
                               seeds=[0, 444, 123, 999, 85, 4444, 64, 128, 808, 838])

    circle_expert.run_expert_sim(env=env, get_from_file=False,
                                 episode_split=[10, 10],
                                 expert_episodes=100, mode="demo", demo_pi=circle_policy,
                                 seeds=[0, 444, 123, 999, 85, 4444, 64, 128, 808, 838])
    circle_memory = circle_expert.memory
    print("Replay Buffers Created")


# 3. Train
mpi_fork(cpu)
logger_kwargs = setup_logger_kwargs(exp_name, 0)

# Good run: exp_name: memo-marigold-rose: https://wandb.ai/openai-scholars/MEMO/runs/h8755blk?workspace=user-feloundou
memo, c_memories = memo_valor(lambda: gym.make(ENV_NAME),
           seed=seed_config,
           memo_kwargs=memo_kwargs_config,
           annealing_kwargs=dict(start=0., stop=1., n_cycle=1, ratio=0.5),
           # episodes_per_epoch=100,   # fix reward accumulation
           episodes_per_epoch=episodes_per_epoch_config,   # fix reward accumulation
           max_ep_len=ep_len_config,
           epochs=5,
           warmup=500,
           train_iters=50,
           # memo_lr=1e-5,  # ideal
           memo_lr=3e-5,  # ideal 3e-5 works well
           # memo_lr=5e-5,  # Learning rate appears to be the most important hyperparameter right now. The smaller the better.
           # memo_lr=3e-4,   # Karpathy Konstant
           train_batch_size=train_batch_size_config,
           eval_batch_size=eval_batch_size_config,
           logger_kwargs=logger_kwargs,
           memories=[marigold_expert.memory, rose_expert.memory,
                     circle_expert.memory])


# Write some assertion code to check determinism in each tests.
# -> Fixed number of experts, what happens if
# the number of experts is greater than or less than number of modes?

# 4. Evaluate
memo_full_eval(model=memo, expert_names=['marigold', 'rose', 'circle'], collated_memories=c_memories, latent_modes=latent_modes_config,
               eval_modes=['class', 'policy', 'quantitative'],
               episodes_per_epoch=episodes_per_epoch_config,
               N_expert=episodes_per_epoch_config*ep_len_config,
               eval_batch_size=eval_batch_size_config, seed=seed_config,
               memo_kwargs=memo_kwargs_config, logger_kwargs=logger_kwargs)

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