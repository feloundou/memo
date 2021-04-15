import gym
import torch
import os.path as osp
from memo.models.neural_nets import MLPActorCritic
from memo.utils.agent_utils import Expert
from memo.utils.buffer_torch import MemoryBatch
from memo.algos.demo_policies import square_policy, forward_policy
from memo.utils.utils import memo_full_eval, setup_logger_kwargs

# 0. Load pre-trained MEMO model
ENV_NAME = 'Safexp-PointGoal1-v0'
env = gym.make(ENV_NAME)

base_path = '/home/tyna/Documents/memo/memo/data/'
memo_file_name = "2experts-4latents-memo-final-2500-step5"
latent_modes_config=5


fname = osp.join(osp.join(base_path, memo_file_name, memo_file_name + '_s0/'), 'pyt_save', 'model' + '.pt')
memo = torch.load(fname)
memo_model = memo[0]

# 1. Initialize Experts
fetch_data=True
forward_expert = Expert(config_name='forward', extension='-policy',
                       record_samples=True, actor_critic=MLPActorCritic,
                       ac_kwargs=dict(hidden_sizes=[128]*4), seed=0)

circle_expert = Expert(config_name='circle', extension='-policy',
                        record_samples=True, actor_critic=MLPActorCritic,
                        ac_kwargs=dict(hidden_sizes=[128]*4), seed=0)

#2. Get expert data
circle_expert.run_expert_sim(env=env, get_from_file=fetch_data,
                             episode_split=[10, 10],
                             expert_episodes=100,mode="demo", replay_buffer_size=10000,
                             seeds=[0, 444, 123, 999, 85, 4444, 64, 128, 808, 838])

forward_expert.run_expert_sim(env=env, get_from_file=fetch_data,
                             episode_split=[10, 10],
                             expert_episodes=100,  mode="demo", demo_pi=forward_policy,
                             seeds=[0, 444, 123, 999, 85, 4444, 64, 128, 808, 838])

# Collate memories
demo_memories = MemoryBatch([circle_expert.memory, forward_expert.memory], step=5)

print("memories", demo_memories)
_, _, _, _ = demo_memories.collate()

#
episodes_per_epoch_config=100
train_batch_size_config=500
eval_batch_size_config=100
ep_len_config=1000
logger_kwargs = setup_logger_kwargs(memo_file_name, 0)


memo_full_eval(model=memo_model, expert_names=['circle', 'forward'],
               file_names=[circle_expert.file_name, forward_expert.file_name],
               pi_types=['demo', 'demo'],
               collated_memories=demo_memories, latent_modes=latent_modes_config,
               # eval_modes=['class', 'policy', 'quantitative'],
               eval_modes=['class', 'policy'],
               episodes_per_epoch=1000, quant_episodes=10,
               N_expert=episodes_per_epoch_config*ep_len_config,
               eval_batch_size=100, seed=0,
               logger_kwargs=logger_kwargs, logging='init')
