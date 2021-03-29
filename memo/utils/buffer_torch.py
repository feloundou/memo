import torch
import numpy as np


class Trajectory:
    def __init__(self):
        self.observations = []
        self.next_observations = []
        self.obs_diff = []
        self.actions = []
        self.rewards = []
        self.costs = []
        self.done = False

    def __len__(self):
        return len(self.observations)


class Buffer:
    def __init__(self, trajectories):
        self.trajectories = trajectories

    def sample(self, next=False):
        # print("traj actions", self.trajectories[0].actions)
        observations_diff = torch.cat([torch.stack(trajectory.obs_diff) for trajectory in self.trajectories])
        observations = torch.cat([torch.stack(trajectory.observations) for trajectory in self.trajectories])
        actions = torch.cat([torch.stack(trajectory.actions) for trajectory in self.trajectories])
        rewards = torch.cat([torch.tensor(trajectory.rewards) for trajectory in self.trajectories])
        costs = torch.cat([torch.tensor(trajectory.costs) for trajectory in self.trajectories])

        next_observations = torch.cat([torch.stack(trajectory.next_observations) for trajectory in self.trajectories])

        if next:
            return observations, actions, rewards, costs, next_observations, observations_diff
        else:
            return observations, actions, rewards, costs

    def __getitem__(self, i):
        return self.trajectories[i]


class MemoryBatch:
    def __init__(self, memories):
        super(MemoryBatch, self).__init__()
        self.idx = 0
        self.memories = memories
        self.size = len(memories)
        self.transition_states = None
        self.transition_actions = None
        self.pure_expert_states = None


    def collate(self):
        '''
        Collates trajectories/episodes/memories from different experts
        :return:
        '''

        for k in range(self.size):
            print("collating memories of size: ", self.size)
            expert_states, expert_actions, _, expert_costs, expert_next_states, _ = self.memories[k].sample(next=True)
            print("Episode costs: ", torch.cumsum(expert_costs, dim=-1))
            print("Memory size: ", expert_states.shape)

            # Exclude the last step of each episode to calculate state differences
            t_states = torch.stack(
                [expert_next_states[i] - expert_states[i] for episode in self.memories[k] for i in
                 range(len(episode) - 1)])
            t_actions = torch.stack(
                [expert_actions[i] for episode in self.memories[k] for i in range(len(episode) - 1)])

            # Three basic checks
            assert t_states.shape[0] == t_actions.shape[
                0], "Tensors for state transitions and actions should be same on dim 0"
            assert torch.equal(expert_next_states[0],
                               expert_states[1]), "The i+1 state tensors should match the i next_state tensors"
            assert torch.equal(expert_states[1] - expert_states[0],
                               t_states[0]), "Check your transition calculations"

            if self.transition_states is None:
                self.transition_states, self.transition_actions = t_states, t_actions
                self.pure_expert_states = expert_states
                # self.expert_ids = torch.empty(self.transition_states.shape[0]).fill_(self.idx)
                self.expert_ids = torch.empty(t_states.shape[0]).fill_(self.idx)

            else:
                self.transition_states = torch.cat([self.transition_states, t_states])
                self.transition_actions = torch.cat([self.transition_actions, t_actions])
                self.pure_expert_states = torch.cat([self.pure_expert_states, expert_states])

                # e_ids = torch.empty(self.transition_states.shape[0]).fill_(self.idx)
                e_ids = torch.empty(t_states.shape[0]).fill_(self.idx)
                self.expert_ids = torch.cat([self.expert_ids, e_ids])

            self.idx += 1

        # print("Final expert ids: ", self.expert_ids)

        return self.transition_states, self.pure_expert_states, self.transition_actions, self.expert_ids


    def eval_batch(self, N_expert, eval_batch_size, episodes_per_epoch):
        # evaluation batch, randomized
        eval_batch_index = None

        for i in range(len(self.memories)):
            curb_factor = episodes_per_epoch
            win_low = i * (N_expert - curb_factor)
            win_high = (i + 1) * (N_expert - curb_factor)

            b_index = torch.randint(low=win_low, high=win_high, size=(eval_batch_size,))

            if eval_batch_index is None:
                eval_batch_index = b_index
            else:
                eval_batch_index = torch.cat([eval_batch_index, b_index])

        eval_raw_states_batch, eval_delta_states_batch, eval_actions_batch, eval_sampled_experts = \
            self.pure_expert_states[eval_batch_index], self.transition_states[eval_batch_index], self.transition_actions[eval_batch_index], \
            self.expert_ids[
                eval_batch_index]
        return eval_raw_states_batch, eval_delta_states_batch, eval_actions_batch, eval_sampled_experts



