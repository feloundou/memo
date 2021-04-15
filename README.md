---

<div align="center">    
 
# MEMO

[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)

<!--
ARXIV   
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)
-->



<!--  
Conference   
-->   
</div>
 
## Description   
Multiple Experts, Multiple Objectives

## How to run   
First, install dependencies   
```bash
# clone memo   
git clone https://github.com/feloundo/memo

# install memo   
cd memo
pip install -e .   
pip install -r requirements.txt
 ```   
 Next, navigate to any file and run it.   
 ```bash
# module folder
cd memo

# run module
python run_memo_experts.py   
```

## Imports
This project is setup as a package which means you can now easily import any file into any other file like so:

```python
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

```

### Examples
For demonstration purposes, I generate generic circle and forward moving agents, and k=4:

The corresponding Weights and Biases run can be found here: https://wandb.ai/openai-scholars/MEMO/runs/2iwn428m?workspace=user-feloundou

You may train any custom agent using the usual methods (e.g. PPO, TRPO, etc...), then evaluate its path. One such sample script for a simple PPO expert can be found in **train_demo_experts.py**.

Some tips:
1. Experiment with the learning rate
2. Experiment with network size
3. Experiment with k (embedding dimension)

### Citation   
```
@article{Eloundou, Florentine,
  title={Multiple Experts, Multiple Objectives},
  author={Your team},
  project={OpenAI Scholars Project},
  year={2021}
}
```   
