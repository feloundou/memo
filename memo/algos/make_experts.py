from memo.utils.agent_utils import Expert
from adabelief_pytorch import AdaBelief

import gym
import safety_gym

from memo.models.ppo_algos import MLPActorCritic
from memo.utils.utils import setup_logger_kwargs, mpi_fork


ENV_NAME = 'Safexp-PointGoal1-v0'


# # make expert
# expert = Expert(config_name='wonder',
#                 record_samples=True,
#                 actor_critic=MLPActorCritic,
#                 ac_kwargs=dict(hidden_sizes=[128] * 4),
#                 seed=0,
#                 penalty_init=5e-3)
#
#     mpi_fork(10)  # run parallel code with mpi
#
#     CONFIG_LIST2 = CONFIG_LIST
#
#     for configuration in CONFIG_LIST2:
#         print(configuration)
#
#         logger_kwargs_BIG = setup_logger_kwargs(configuration['name'], configuration['seed'])
#
#         BIG_EXPERT = Expert(config_name=configuration['name'],
#                             record_samples=True,
#                             actor_critic=MLPActorCritic,
#                             ac_kwargs=dict(hidden_sizes=[configuration['hid']] * configuration['l']),
#                             seed=configuration['seed'],
#                             penalty_init=5e-3)
#
#
#         BIG_EXPERT.ppo_train(env_fn=lambda: gym.make(ENV_NAME),
#                              epochs=500,
#                              gamma=configuration['gamma'],
#                              lam=configuration['lam'],
#                              steps_per_epoch=configuration['steps'],
#                              train_pi_iters=100,
#                              pi_lr=3e-4,
#                              train_vf_iters=100,
#                              vf_lr=1e-3,
#                              penalty_lr=configuration['penalty_lr'],
#                              cost_lim=configuration['cost_lim'],
#                              clip_ratio=0.2,
#                              max_ep_len=1000,
#                              save_every=10,
#                              wandb_write=False,
#                              logger_kwargs=logger_kwargs_BIG)
#
#         print("just finished!")
# logger_kwargs = setup_logger_kwargs('STANDARDTEST', 0)

#
# expert.ppo_train(env_fn=lambda: gym.make(ENV_NAME),
#                  epochs=50,
#                  gamma=0.99,
#                  lam=0.98,
#                  steps_per_epoch=5000,
#                  train_pi_iters=100,
#                  pi_lr=3e-4,
#                  train_vf_iters=100,
#                  vf_lr=1e-3,
#                  penalty_lr=5e-3,
#                  cost_lim=10,
#                  clip_ratio=0.2 ,
#                  max_ep_len=1000,
#                  save_every=10,
#                  wandb_write=False,
#                  logger_kwargs=logger_kwargs)

########################################################################################################################
########################################################################################################################

exec(open('../config/nn_config.py').read(), globals())

# print(standard_config)
mpi_fork(2)  # run parallel code with mpi

CONFIG_LIST2 = CONFIG_LIST

for configuration in CONFIG_LIST2:
    print(configuration)


    logger_kwargs_BIG = setup_logger_kwargs(configuration['name'], configuration['seed'])


    BIG_EXPERT = Expert(config_name=configuration['name'],
                        record_samples=True,
                        actor_critic=MLPActorCritic,
                        ac_kwargs=dict(hidden_sizes=[configuration['hid']] * configuration['l']),
                        seed=configuration['seed'],
                        penalty_init=5e-3)


    BIG_EXPERT.ppo_train(env_fn=lambda: gym.make(ENV_NAME),
                         epochs=1000,
                         gamma=configuration['gamma'],
                         lam=configuration['lam'],
                         steps_per_epoch=configuration['steps'],
                         train_pi_iters=100,
                         pi_lr=3e-4,
                         train_vf_iters=100,
                         vf_lr=1e-3,
                         penalty_lr=configuration['penalty_lr'],
                         cost_lim=configuration['cost_lim'],
                         clip_ratio=0.2,
                         max_ep_len=1000,
                         save_every=10,
                         wandb_write=False,
                         logger_kwargs=logger_kwargs_BIG)

    print("just finished!")






