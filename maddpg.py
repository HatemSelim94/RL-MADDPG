import numpy as np
from collections import deque
import torch
from maddpg_agent import *

class MADDPG_Coordinator:
    """
       :agents: DDPG agents
       :agents_num: number of DDPG agents
    """
    def __init__(self, agents, agents_num, memory, update_every = 20):
        self.agents = agents
        self.agents_num = agents_num
        self.memory = memory
        self.t_step = 0
        self.update_every = update_every
    
    def reset(self):
        for agent in self.agents:
            agent.reset()
    
    def act(self, states, add_noise=True):
        actions = []
        for i in range(self.agents_num):
            action = self.agents[i].act(states[i], add_noise)
            actions.append(action)
        actions = np.asarray(actions)
        return actions
    
    def step(self, states, actions, rewards, next_states, dones):
        self.memory.add(states, actions, rewards, next_states, dones)
        if len(self.memory)>10000:
            self.t_step = (self.t_step + 1) % self.update_every
            if self.t_step == 0:
                for agent_id in range(self.agents_num):
                    experiences = self.memory.sample()
                    self.agents[agent_id].step(experiences, agent_id)
    
    def save(self, i_episode):
        for agent_id in range(self.agents_num):
            actor_training_state = {'episode': i_episode, 
                              'agent_actor_dict': self.agents[agent_id].actor_local.state_dict(),
                              'agent_actor_optimizer':self.agents[agent_id].actor_optimizer.state_dict()}
            critic_training_state = {'episode': i_episode, 
                              'agent_critic_dict': self.agents[agent_id].critic_local.state_dict(),
                              'agent_critic_optimizer':self.agents[agent_id].critic_optimizer.state_dict()}
            torch.save(actor_training_state, 'agent{}_checkpoint_actor.pth'.format(agent_id))
            torch.save(critic_training_state, 'agent{}_checkpoint_critic.pth'.format(agent_id))


def maddpg_train(agents_num, action_size, state_size, env, brain_name,buffer_size = int(2e5),batch_size = 512, seed = 0, n_episodes=5000, max_t=300000, print_every=100, update_every = 20, tau = 0.0005, actorLr =1e-4, criticLr = 1e-4, weight_decay = 0):
    """
      :agents (DDPG agent): DDPG agents
      :env (Unity environment): Unity environment(Reacher 1 agent)
      :brain_name: environment name
      :action_size: action space size per agent
      :state_size: state space size per agent
      :n_episodes (int): Number of training episodes
      :max_t (int): Max. steps per episode
      :print_every(int): Frequncy of printing the avg. score
    """
    scores = []
    avg_scores = []
    scores_deque = deque(maxlen=print_every)
    memory = ReplayBuffer(action_size, buffer_size, batch_size, agents_num,seed)
    agents = [Agent(state_size, action_size,batch_size=batch_size, buffer_size = buffer_size,tau = tau, actorLr =actorLr, criticLr = criticLr, weight_decay = weight_decay ) for _ in range(agents_num)]
    maddpg_coordinator = MADDPG_Coordinator(agents, agents_num, memory, update_every = update_every)
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        maddpg_coordinator.reset()
        score = np.zeros(agents_num)
        add_noise = True
        for t in range(max_t):
            if i_episode >= 5000:
                add_noise = False
            actions = maddpg_coordinator.act(states, add_noise)
            env_info = env.step(actions)[brain_name] 
            next_states = env_info.vector_observations  
            rewards = env_info.rewards
            dones = env_info.local_done
            maddpg_coordinator.step(states, actions, rewards, next_states, dones)
            states = next_states
            score += rewards
            if np.any(dones):
                break
        eps_score = np.max(score)
        scores_deque.append(eps_score)
        scores.append(eps_score) 
        print('\rEpisode {}\t Episode score: {:.2f}'.format(i_episode, eps_score), end="")
        if i_episode % print_every == 0:
            avg_score = np.mean(scores_deque)
            avg_scores.append(avg_score)
            print('\rEpisode {}\tAverage Score in 100 episodes: {:.2f}\t Episode score: {}'.format(i_episode, avg_score, eps_score))
            print(len(memory))
            if avg_score >= 0.5:
                maddpg_coordinator.save(i_episode)
                break
    return scores, avg_scores