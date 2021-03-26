import gym
from memo.models.neural_nets import MLPActorCritic
from memo.utils.agent_utils import Expert
from memo_valor import memo_valor
from memo.utils.utils import setup_logger_kwargs, run_learner_eval


# Run MEMO

ENV_NAME = 'Safexp-PointGoal1-v0'

# make environment
env = gym.make(ENV_NAME)

#1. Make Experts
marigold_expert = Expert(config_name='marigold',
                record_samples=True, actor_critic=MLPActorCritic,
                ac_kwargs=dict(hidden_sizes=[128] * 4), seed=0) #444

# Seed here helps reset initialization for episodes for the perfect set of trajectories
rose_expert = Expert(config_name='rose',
                record_samples=True,  actor_critic=MLPActorCritic,
                ac_kwargs=dict(hidden_sizes=[128] * 4), seed=123)   # 123


# 2. Make the dataset
# marigold
marigold_expert.run_expert_sim(env=env, get_from_file=True, expert_episodes=100, replay_buffer_size=10000)
marigold_rb = marigold_expert.replay_buffer
marigold_memory = marigold_expert.memory

# rose
rose_expert.run_expert_sim(env=env, get_from_file=True, expert_episodes=100, replay_buffer_size=10000)
rose_rb = rose_expert.replay_buffer
rose_memory = rose_expert.memory

# # print("replay buffers fetched")
# # Run simulation to collect demonstrations (just a gut-check here)
# marigold_expert.run_expert_sim(env=env, get_from_file=False,
#                                max_cost=200, min_reward=20, episode_split=[10, 10],
#                                expert_episodes=100,  replay_buffer_size=10000,
#                                seeds=[0, 444, 123, 999, 85, 4444, 64, 128, 808, 838])  ## TOD0: change back to 15
# marigold_rb = marigold_expert.replay_buffer
# marigold_memory = marigold_expert.memory
#
# rose_expert.run_expert_sim(env=env, get_from_file=False, max_cost=10, min_reward=-10, episode_split=[10, 10],
#                            expert_episodes=100, replay_buffer_size=10000,
#                            seeds=[0, 444, 123, 999, 85, 4444, 64, 128, 808, 838])
# rose_rb = rose_expert.replay_buffer
# rose_memory = rose_expert.memory

ep_len_config = 1000

# 3. Train
logger_kwargs = setup_logger_kwargs('memo-vae-newest', 0)

# Good run: exp_name: memo-marigold-rose: https://wandb.ai/openai-scholars/MEMO/runs/h8755blk?workspace=user-feloundou
memo_valor(lambda: gym.make(ENV_NAME),
           seed=0,
           # seed=123,  # some intuition.
           vaelor_kwargs=dict(encoder_hidden=[1000],
                              decoder_hidden=[200]),
           annealing_kwargs=dict(start=0., stop=1., n_cycle=1, ratio=0.5),
           episodes_per_epoch=100,   # fix reward accumulation
           max_ep_len=ep_len_config,
           epochs=50,
           train_valor_iters=25,
           vae_lr=1e-5,  # ideal
           # vae_lr=5e-5,  # Learning rate appears to be the most important hyperparameter right now. The smaller the better.
           # vae_lr=3e-4,   # Karpathy Konstant   (apparently too high for my needs currently)
           train_batch_size=50,
           eval_batch_size=100,
           logger_kwargs=logger_kwargs,
           replay_buffers=[marigold_rb, rose_rb],
           memories=[marigold_memory, rose_memory])

# Some tips to train MEMO:
# 1. # Warmup improves it
# 2. # Smaller training rates are better
# 3. # Generate as much data as your computer can handle.