# https://spinningup.openai.com/en/latest/algorithms/ddpg.html
# https://github.com/rll/rllab
# https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch/blob/master/agents/actor_critic_agents/DDPG.py
# https://math.stackexchange.com/q/1288406

import numpy as np
import random
import copy
from collections import namedtuple, deque

from models import ActorNetwork, CriticNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """DDPG Agent"""
    def __init__(self, state_size, action_size, memory=None, hd1_units=400, hd2_units=300 ,random_seed = 0, buffer_size = int(2e5), batch_size = 256, tau = 0.0005, actorLr =1e-3, criticLr = 1e-3, weight_decay = 0, update_every = 20, gamma = 0.999):
        """ :state_size (int): dimension of each state
            :action_size (int): dimension of each action
            :hd1_units (int) : number of the first hidden layer units
            :hd1_units (int) : number of the second hidden layer units
            :random_seed (int): random seed
            :buffer_size (int): replay buffer size
            :batch_size (int): batch size
            :tau (float): interpolation factor
            :actorLr (float): actor learning rate
            :criticLr (float): critic learning rate
            :weight_decay (float): Optimizer L2 penalty
            :update_every (int): learning frequency
            :gamma (float): Discount factor
        """
        self.state_size = state_size
        self.action_size = action_size
        self.update_every = update_every
        self.gamma = gamma
        self.tau = tau
        self.memory= memory
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        random.seed(random_seed)

        # Actor & Target Networks
        self.actor_local = ActorNetwork(state_size, action_size, random_seed).to(device)
        self.actor_target = ActorNetwork(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=actorLr, weight_decay = weight_decay)

        # Critic & Target Networks
        self.critic_local = CriticNetwork(state_size, action_size, random_seed).to(device)
        self.critic_target = CriticNetwork(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=criticLr, weight_decay=weight_decay)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)
        self.t_step = 0

    def step(self, experiences, agent_id):
        self.learn(experiences,self.gamma, agent_id)

    def act(self, state, add_noise=True):
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        # manual action clipping
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma, agent_id):
        """
            :experiences (Tuple): Transition parameters (s, a, r, s', done)
            :gamma (float): Discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        # Update critic
        # Get the predicted next-state actions and Q values from target nets
        with torch.no_grad():
            actions_next = self.actor_target(next_states)
            #print("actions next shape:{}".format(actions_next.shape))
            next_states = next_states.reshape(self.batch_size, -1)
            actions_next = actions_next.reshape(self.batch_size, -1)
            Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        #print("Q targets next shape {} dones {} rew{}".format(Q_targets_next.shape, dones[:,agent_id].shape, rewards[:,agent_id].shape))
        dones_local = dones[:,agent_id].reshape(self.batch_size, -1)
        rewards_local = rewards[:,agent_id].reshape(self.batch_size, -1)
        Q_targets = rewards_local + (gamma * Q_targets_next * (1 - dones_local))
        #print("Q targets shape{}".format(Q_targets.shape))
        # Compute critic loss
        global_states = states.reshape(self.batch_size, -1)
        global_actions = actions.reshape(self.batch_size, -1)
        Q_expected = self.critic_local(global_states, global_actions)
        #print("Q expected shape{}".format(Q_expected.shape))
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 5)
        self.critic_optimizer.step()

        # Update Actor
        # Compute actor loss
        #debug_func(states)
        local_states = states[:,agent_id,:]
        actions_pred = self.actor_local(local_states)
        global_states = states.reshape(self.batch_size, -1) 
        global_actions = actions
        global_actions[:,agent_id,:] = actions_pred
        global_actions = global_actions.reshape(self.batch_size,-1)
        #debug_func(global_states, global_actions)
        actor_loss = -self.critic_local(global_states, global_actions).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 5)
        self.actor_optimizer.step()

        # update targets
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """
            local_model: Source
            target_model: Destination
            tau (float):  Interpolation factor
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process"""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        random.seed(seed)
        self.x_prev = None
        self.reset()

    def reset(self):
        self.x_prev = copy.copy(self.mu)

    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) + self.sigma * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x


class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, agents_num,seed):
        """ 
            :buffer_size (int): Max. buffer size
            :batch_size (int): Batch size
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.agents_num = agents_num
        random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.stack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.stack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.stack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        #debug_func(states, actions, rewards, next_states, dones)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of replay buffer."""
        return len(self.memory)

def debug_func(*args):
    for arg in args:
        print(arg.shape)
    print("----end----")