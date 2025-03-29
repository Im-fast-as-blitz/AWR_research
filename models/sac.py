import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal
from utils import ReplayBuffer
from random_actor import RandomActor
from utils import play_and_record, update_target_networks, optimize
from losses import SAC_loss, compute_actor_loss_sac


class Critic_SAC(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size

        self.network = nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, 1),
            nn.Flatten(0)
        )

    def get_qvalues(self, states, actions):
        '''
        input:
            states - tensor, (batch_size x features)
            actions - tensor, (batch_size x actions_dim)
        output:
            qvalues - tensor, critic estimation, (batch_size)
        '''
        qvalues = self.network(torch.concat([states, actions], dim=1))

        assert len(qvalues.shape) == 1 and qvalues.shape[0] == states.shape[0]

        return qvalues
    

class Actor_SAC(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size

        self.network = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, 2 * self.action_dim)
        )

    def apply(self, states):
        '''
        For given batch of states samples actions and also returns its log prob.
        input:
            states - PyTorch tensor, (batch_size x features)
        output:
            actions - PyTorch tensor, (batch_size x action_dim)
            log_prob - PyTorch tensor, (batch_size)
        '''
        mean, var = self.network(states).split(self.action_dim, dim=-1)
        var = nn.Softplus()(var)
        m = Normal(mean, var)
        sample = m.rsample()
        actions = torch.tanh(sample)
        log_prob = m.log_prob(sample) - torch.log((1 - actions**2).clip(min=1e-9)).sum(dim=-1, keepdim=True)
        return actions, log_prob.sum(dim=-1)

    def get_action(self, states):
        '''
        Used to interact with environment by sampling actions from policy
        input:
            states - numpy, (batch_size x features)
        output:
            actions - numpy, (batch_size x actions_dim)
        '''
        with torch.no_grad():
            actions = self.apply(states)[0].cpu().numpy()

            assert isinstance(actions, (list,np.ndarray)), "convert actions to numpy to send into env"
            assert actions.max() <= 1. and actions.min() >= -1, "actions must be in the range [-1, 1]"
            return actions


class SAC(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, buffer_size, device, *args, **kwargs):
        super().__init__()

        self.actor = Actor_SAC(state_dim, action_dim, hidden_size).to(device)
        self.critic_1 = Critic_SAC(state_dim, action_dim, hidden_size).to(device)
        self.critic_2 = Critic_SAC(state_dim, action_dim, hidden_size).to(device)
        self.target_critic_1 = Critic_SAC(state_dim, action_dim, hidden_size).to(device)
        self.target_critic_2 = Critic_SAC(state_dim, action_dim, hidden_size).to(device)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        lr = kwargs['lr'] if kwargs['lr'] is not None else 3e-4
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.optimizer_critic_1 = torch.optim.Adam(self.critic_1.parameters(), lr=lr)
        self.optimizer_critic_2 = torch.optim.Adam(self.critic_2.parameters(), lr=lr)
        self.random_actor = RandomActor().to(self.device)
        self.exp_replay = ReplayBuffer(buffer_size)
        self.device = device
        self.start_timesteps = kwargs['start_timesteps'] if kwargs['start_timesteps'] is not None else 5000
        self.timesteps_per_epoch = kwargs['timesteps_per_epoch'] if kwargs['timesteps_per_epoch'] is not None else 1
        self.batch_size = kwargs['batch_size'] if kwargs['batch_size'] is not None else 128
        self.loss = SAC_loss(kwargs['alpha'] if kwargs['alpha'] is not None else 0.1)


    def run_iter(self, *args, **kwargs):
        env = kwargs['env']
        if len(self.exp_replay) < self.start_timesteps:
            _, interaction_state = play_and_record(interaction_state, self.random_actor, env, self.exp_replay, self.timesteps_per_epoch)
            return

        # perform a step in environment and store it in experience replay
        _, interaction_state = play_and_record(interaction_state, self.actor, env, self.exp_replay, self.timesteps_per_epoch)

        # sample a batch from experience replay
        states, actions, rewards, next_states, is_done = self.exp_replay.sample(self.batch_size)

        # move everything to PyTorch tensors
        states = torch.tensor(states, device=self.device, dtype=torch.float)
        actions = torch.tensor(actions, device=self.device, dtype=torch.float)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float)
        next_states = torch.tensor(next_states, device=self.device, dtype=torch.float)
        is_done = torch.tensor(
            is_done.astype('float32'),
            device=self.device,
            dtype=torch.float
        )

        # losses
        q_values_1 = self.critic_1.get_qvalues(states, actions)
        q_values_2 = self.critic_2.get_qvalues(states, actions)
        loss_1, loss_2 = self.loss(q_values_1, q_values_2)
        # optimize("critic1", self.critic_1, self.optimizer_critic_1, critic1_loss)

        # critic2_loss = mse(critic_target, critic2.get_qvalues(states, actions))
        # optimize("critic2", self.critic_2, self.optimizer_critic_2, critic2_loss)

        # if n_iterations % policy_update_freq == 0:
        #     actor_loss = compute_actor_loss(states, alpha)
        #     optimize("actor", self.actor, self.optimizer_actor, actor_loss)

        #     # update target networks
        #     update_target_networks(critic1, target_critic1)
        #     update_target_networks(critic2, target_critic2)