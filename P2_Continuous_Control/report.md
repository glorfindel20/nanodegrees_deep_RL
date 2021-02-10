# nanodegrees_deep_RL
# Project 2: Continuous Control

### Algorithm
In order to solve this challenge, I have explored and implemented the Deep Deterministic Policy Gradient algorithm (DDPG), as described in this paper: [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971).

# Included in this repository
The code used to create and train the Agent
Continuous_Control.ipynb
ddpg_agent.py 
ddpg_agent_PER_slow.py
ddpg_agent_PER.py
model.py
The trained model
checkpoint_ddpg.pt
A file describing all the packages required to set up the environment
environment.yml
A Report.md file describing the development process and the learning algorithm, along with ideas for future work
This README.md file

### Development

For this project I tried different configurations before finding the optimal one.

One of my goals was to successfully implement PER: Prioritized Experience Replay.
I implement this in ddpg_agent_PER_slow.py, but training was very very slow. 
This is why I gave up using it, and I chose to optimize the DDPG.


### Fine-tuning the hyperparameters

I have tried many configurations:
  - Network size: i tried with [128,128] , [64,128] , [128,256] but in the end i used the layout suggested in the paper: two hidden layers, the first with 400 nodes and the second with 300 nodes, for both Actor and Critic networks.
  - BUFFER_SIZE: I start from 1e4 to 1e5 and endig with 1e6.
  - BATCH_SIZE: start from 128, try with 256 and in the end used 64

The final used hyperparameters:

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 64      # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay

GRAD_CLIPPING = 1.0     # Gradient Clipping
EPSILON = 1.0     # for epsilon in the noise process (act step)
EPSILON_DECAY = 1e-6


## Result


## Ideas for Future Work

 1) use succesfully PER - Prioritized Experience Replay.
 2) test A3C - Asynchronous Advantage Actor-Critic.
