### Environment
Two rackets, represented by two agents, play against each other. If a racket (an agent) hits the ball over the net, it receives a reward of +0.1. If a racket(an agent) lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

After each episode, the rewards that each agent received (without discounting) are added, to get a score for each agent. This yields two (potentially different) scores. We then take the maximum of these two scores. This yields a single score for each episode.
The problem is considered to be solved when the average of the scores over 100 consecutive episodes is larger than 0.5.

More info about the solution can be found [here](https://github.com/HatemSelim94/RL-MADDPG/blob/main/Report.md).

#### Dependencies
##### Activate the environment 
To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
    	```bash
    	conda create --name drlnd python=3.6
    	source activate drlnd
    	```
	- __Windows__: 
    	```bash
    	conda create --name drlnd python=3.6 
    	activate drlnd
    	```
	
2. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.  
	- Next, install the **classic control** environment group by following the instructions [here](https://github.com/openai/gym#classic-control).
	- Then, install the **box2d** environment group by following the instructions [here](https://github.com/openai/gym#box2d).
	
3. Clone the repository (if you haven't already!), and navigate to the `python/` folder.  Then, install several dependencies.
    ```bash
    git clone https://github.com/udacity/deep-reinforcement-learning.git
    cd deep-reinforcement-learning/python
    pip install .
    ```

4. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.  
    ```bash
    python -m ipykernel install --user --name drlnd --display-name "drlnd"
    ```

5. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu. 
##### Download the environment

* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
* Windows 64: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    Then, place the file in the p3_collab-compet/ folder in the DRLND GitHub repository, and unzip (or decompress) the file.
##### Train the agents
* Run Tennis.ipynb or run Tennis. py (Do not forget to activate conda environment first)

* Indisde Tennis file, maddpg_train function is called to start training:
    ```python
    from maddpg import maddpg_train
    scores, avg_scores =  maddpg_train(agents_num, action_size, state_size, env,  brain_name,buffer_size = int(1e5),batch_size = 256, seed = 0, n_episodes=50000,  max_t=4000, print_every=100, update_every = 4, tau = 1e-3, actorLr =1e-4, criticLr = 1e-3, weight_decay = 0)
    ```

