import time
import joblib
import os
import os.path as osp
import gym
from cpprb import ReplayBuffer
import pickle
import wandb
import numpy as np
import torch

from memo.utils.utils import EpochLogger, num_procs


def load_policy_and_env(fpath, itr='last', deterministic=False, type='ppo'):
    """
    Load a policy from save along with RL env.
    Not exceptionally future-proof, but it will suffice for basic uses of the
    Spinning Up implementations.
    loads as if there's a PyTorch save.
    """

    # handle which epoch to load from
    if itr == 'last':
        # check filenames for epoch (AKA iteration) numbers, find maximum value

        pytsave_path = osp.join(fpath, 'pyt_save')
        # Each file in this folder has naming convention 'modelXX.pt', where
        # 'XX' is either an integer or empty string. Empty string case
        # corresponds to len(x)==8, hence that case is excluded.
        saves = [int(x.split('.')[0][5:]) for x in os.listdir(pytsave_path) if len(x) > 8 and 'model' in x]

        itr = '%d' % max(saves) if len(saves) > 0 else ''

    else:
        assert isinstance(itr, int), \
            "Bad value provided for itr (needs to be int or 'last')."
        itr = '%d' % itr

    # load the get_action function
    get_action = load_pytorch_policy(fpath, itr, deterministic, type=type)

    # try to load environment from save
    # (sometimes this will fail because the environment could not be pickled)
    try:

        print(osp.join(fpath, 'vars' + itr + '.pkl'))
        state = joblib.load(osp.join(fpath, 'vars' + itr + '.pkl'))
        env = state['env']
    except:
        env = None

    return env, get_action


def load_pytorch_policy(fpath, itr, deterministic=False, type='ppo'):
    """ Load a pytorch policy saved with Spinning Up Logger."""

    fname = osp.join(fpath, 'pyt_save', 'model' + itr + '.pt')
    print('\n\nLoading from %s.\n\n' % fname)

    model = torch.load(fname)
    print("Model Loaded!", model)

    if type=='memo':
        def get_action(x,c):
            with torch.no_grad():
                x = torch.as_tensor(x, dtype=torch.float32)
                vaelor=model[0]
                action = vaelor.act(x, c)
            return action
    else:
        def get_action(x):
            with torch.no_grad():
                x = torch.as_tensor(x, dtype=torch.float32)
                # print("X is: ", x)
                # print("Internal model is: ", model)
                action = model.act(x)

            return action

    # make function for producing an action given a single state
    # investigate accuracy of normal distributions

    return get_action


def run_policy(env, get_action, max_ep_len=None, num_episodes=100, render=True, record=False, record_project='benchmarking', record_name='trained', data_path='', config_name='test', max_len_rb=100, benchmark=False, log_prefix=''):
    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
        "page on Experiment Outputs for how to handle this situation."

    logger = EpochLogger()
    o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
    ep_cost = 0
    local_steps_per_epoch = int(4000 / num_procs())

    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    rew_mov_avg_10 = []
    cost_mov_avg_10 = []

    if benchmark:
        ep_costs = []
        ep_rewards = []

    if record:
        wandb.login()
        # 4 million env interactions
        wandb.init(project=record_project, name=record_name)

        rb = ReplayBuffer(size=10000,
                          env_dict={
                              "obs": {"shape": obs_dim},
                              "act": {"shape": act_dim},
                              "rew": {},
                              "next_obs": {"shape": obs_dim},
                              "done": {}})

    # A = [-1, -1]  # circle    B = [-1, 1]     C = [1, -1]  # circle, pivoting around butt    D = [1, 1]
    E = [-1, 0.25]  # best circle so far [ The smaller y, the larger the circle
    F = [-1, 0] # go forward
    G = [-1, 0]

    circle_policy = [-1, 0.25]
    forward_policy = [-1, 0]

    while n < num_episodes:
        if render:
            env.render()
            time.sleep(1e-3)

        # a = get_action(o)
        a = [1, 0]

        print("action taken: ", a)
        next_o, r, d, info = env.step(a)

        if record:
            # buf.store(next_o, a, r, None, info['cost'], None, None, None)
            done_int = int(d==True)
            rb.add(obs=o, act=a, rew=r, next_obs=next_o, done=done_int)

        ep_ret += r
        ep_len += 1
        ep_cost += info['cost']

        # Important!
        o = next_o

        if d or (ep_len == max_ep_len): # finish recording and save csv
            if record:
                rb.on_episode_end()

                # make directory if does not exist
                if not os.path.exists(data_path + config_name + '_episodes'):
                    os.makedirs(data_path + config_name + '_episodes')

            if len(rew_mov_avg_10) >= 25:
                rew_mov_avg_10.pop(0)
                cost_mov_avg_10.pop(0)

            rew_mov_avg_10.append(ep_ret)
            cost_mov_avg_10.append(ep_cost)

            mov_avg_ret = np.mean(rew_mov_avg_10)
            mov_avg_cost = np.mean(cost_mov_avg_10)

            expert_metrics = {log_prefix + 'episode return': ep_ret,
                              log_prefix + 'episode cost': ep_cost,
                              log_prefix + '25ep mov avg return': mov_avg_ret,
                              log_prefix + '25ep mov avg cost': mov_avg_cost
                              }

            if benchmark:
                ep_rewards.append(ep_ret)
                ep_costs.append(ep_cost)

            wandb.log(expert_metrics)
            logger.store(EpRet=ep_ret, EpLen=ep_len, EpCost=ep_cost)
            print('Episode %d \t EpRet %.3f \t EpLen %d \t EpCost %d' % (n, ep_ret, ep_len, ep_cost))
            o, r, d, ep_ret, ep_len, ep_cost = env.reset(), 0, False, 0, 0, 0
            n += 1


    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()

    if record:
        print("saving final buffer")
        bufname_pk = data_path + config_name + '_episodes/sim_data_' + str(int(num_episodes)) + '_buffer.pkl'
        file_pi = open(bufname_pk, 'wb')
        pickle.dump(rb.get_all_transitions(), file_pi)
        wandb.finish()

        return rb

    if benchmark:
        return ep_rewards, ep_costs


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--fpath', type=str,
                        default= '/home/tyna/Documents/openai/research-project/data/ppo_test/ppo_test_s0/')
    parser.add_argument('--len', '-l', type=int, default=0)
    parser.add_argument('--episodes', '-n', type=int, default=100)
    parser.add_argument('--norender', '-nr', action='store_true')
    # parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    args = parser.parse_args()

    file_name = 'ppo_penalized_test'  # second best
    config_name = 'cyan'


    base_path = '/home/tyna/Documents/safe-experts/data/'
    expert_path = '/home/tyna/Documents/openai/research-project/expert_data/'


    _, get_action = load_policy_and_env(osp.join(base_path, file_name, file_name + '_s0/'),
                                        args.itr if args.itr >= 0 else 'last',
                                        args.deterministic)

    print("Get action: ", get_action)

    env = gym.make('Safexp-PointGoal1-v0')
    print("Just made env")
    env._seed = 0
    # run_policy(env, get_action, args.len, args.episodes, not (args.norender), record=False, data_path=base_path, config_name='cyan', max_len_rb)
    # run_policy(env, get_action, args.len, args.episodes, False, record=True, data_path=expert_path, config_name=config_name, max_len_rb=10000)
    run_policy(env, get_action, args.len, args.episodes, True, record=True, data_path=expert_path,
               config_name=config_name, max_len_rb=10000)


