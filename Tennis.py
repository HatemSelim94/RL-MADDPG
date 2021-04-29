from unityagents import UnityEnvironment
from maddpg import maddpg_train
import matplotlib.pyplot as plt
import numpy as np

# Unity environment
env = UnityEnvironment(file_name="/data/Tennis_Linux_NoVis/Tennis")
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]
states = env_info.vector_observations

# Start training
state_size = states.shape[1]
action_size = brain.vector_action_space_size
agents_num=2
scores, avg_scores =  maddpg_train(agents_num, action_size, state_size, env, 
                       brain_name,buffer_size = int(1e5),batch_size = 256, seed = 0, n_episodes=50000, 
                       max_t=4000, print_every=100, update_every = 4, tau = 1e-3, actorLr =1e-4,
                       criticLr = 1e-3, weight_decay = 0)
# Plot result
episodes = list(range(1, len(scores)+1))
plt.plot(episodes, scores,marker='*', label="Score per episode")
plt.plot(np.array(list(range(1,len(avg_scores)+1)))*100,avg_scores, label="Average score over 100 episodes")
plt.title("MADDPG-Tennis")
plt.xlabel("Episode")
plt.ylabel("Score")
plt.axvline(episodes[-1]-100,color='g')
plt.axhline(0.5,color='g')
plt.legend()
plt.savefig('MADDPG_Tennis_score.png', bbox_inches='tight')