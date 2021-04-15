import gym
import memo
from memo.models.neural_nets import MLPActorCritic
from memo.utils.agent_utils import Expert
from memo_valor import memo_valor
from memo.utils.utils import setup_logger_kwargs, mpi_fork
from memo.algos.demo_policies import circle_policy, forward_policy
from memo.utils.utils import memo_full_eval

# Run MEMO
ENV_NAME = 'Safexp-PointGoal1-v0'
exp_name = "forward-circle-experts-4modes"

# lr_config=3e-4
lr_config=3e-5

ep_len_config = 1000
episodes_per_expert_config=100
cpu=1
step_size_config=1

latent_modes_config=4

epochs_config=50
train_batch_size_config=500
eval_batch_size_config=100
seed_config=0

train_iters_config=200

memo_kwargs_config=dict(encoder_hidden=[1000],
                        decoder_hidden=[1000]*2,
                        actor_hidden=[256]*2,
                        latent_modes=latent_modes_config)

# 0. Make environment
env = gym.make(ENV_NAME)

# 1. Make Experts
forward_expert = Expert(config_name='forward', extension='-policy',
                        record_samples=True, actor_critic=MLPActorCritic,
                        ac_kwargs=dict(hidden_sizes=[128]*4), seed=seed_config)

circle_expert = Expert(config_name='circle', extension='-policy',
                record_samples=True,  actor_critic=MLPActorCritic,
                ac_kwargs=dict(hidden_sizes=[128]*4), seed=seed_config)

# 2. Make the dataset (you can train them with expert.ppo_train() but these were pre-trained)
create_data = False

forward_expert.run_expert_sim(env=env, get_from_file=not create_data,
                             episode_split=[10, 10],
                             expert_episodes=100,  mode="demo", demo_pi=forward_policy,
                             seeds=[0, 444, 123, 999, 85, 4444, 64, 128, 808, 838])


circle_expert.run_expert_sim(env=env, get_from_file=not create_data,
                             episode_split=[10, 10],
                             expert_episodes=100, mode="demo", demo_pi=circle_policy,
                             seeds=[0, 444, 123, 999, 85, 4444, 64, 128, 808, 838])
print("Replay Buffers Created")

# 3. Train
mpi_fork(cpu)
logger_kwargs = setup_logger_kwargs(exp_name, seed_config)

#
memo, c_memories = memo_valor(lambda: gym.make(ENV_NAME),
           seed=seed_config,
           memo_kwargs=memo_kwargs_config,
           annealing_kwargs=dict(start=0., stop=1., n_cycle=1, ratio=0.5),
           episodes_per_expert=episodes_per_expert_config,   # fix reward accumulation
           max_ep_len=ep_len_config,
           epochs=epochs_config,
           train_iters=train_iters_config,
           step_size=step_size_config,
           memo_lr=lr_config,
           train_batch_size=train_batch_size_config,
           eval_batch_size=eval_batch_size_config,
           logger_kwargs=logger_kwargs,
           memories=[circle_expert.memory, forward_expert.memory])



# 4. Evaluate

memo_full_eval(model=memo, expert_names=['circle', 'forward'],
               file_names=[circle_expert.file_name, forward_expert.file_name],
               pi_types=['demo', 'demo'],
               collated_memories=c_memories, latent_modes=latent_modes_config,
               # eval_modes=['class', 'policy', 'quantitative'],
               eval_modes=['class', 'policy'],
               episodes_per_epoch=episodes_per_expert_config, quant_episodes=10,
               N_expert=episodes_per_expert_config*ep_len_config,
               eval_batch_size=eval_batch_size_config, seed=seed_config,
               logger_kwargs=logger_kwargs)


# if __name__ == '__main__':
#     import argparse

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--env', type=str, default='Safexp-PointGoal1-v0')
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
