# nanodegrees_deep_RL
Udacity Deep Reinforcement Learning Nanodegree


# Learning Algorithm

1) Deep Q-Network (DQN) + Experience Replay
With Deep Q-Learning, a deep neural network is used to approximate the Q-function. Given a network F, finding an optimal policy is a matter of finding the best weights w such that F(s,a,w) ≈ Q(s,a). [original paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)

The neural network architecture used for this project can be found in the model.py file of the source code. The network contains three fully connected layers with 64, 64, and 4 nodes respectively. Testing of bigger networks (more nodes) and deeper networks (more layers) did not produce better results.

As for the network inputs, rather than feeding-in sequential batches of experience tuples, I randomly sample from a history of experiences using an approach called Experience Replay.

Experience replay allows the RL agent to learn from past experience.

Each experience is stored in a replay buffer as the agent interacts with the environment. The replay buffer contains a collection of experience tuples with the state, action, reward, and next state (s, a, r, s'). The agent then samples from this buffer as part of the learning step. Experiences are sampled randomly, so that the data is uncorrelated. This prevents action values from oscillating or diverging catastrophically, since a naive Q-learning algorithm could otherwise become biased by correlations between sequential experience tuples.

Also, experience replay improves learning through repetition. By doing multiple passes over the data, our agent has multiple opportunities to learn from a single experience tuple. This is particularly useful for state-action pairs that occur infrequently within the environment.

The implementation of the replay buffer can be found in the dqn_agent.py file of the source code.

2) Deep Q-Network (DQN) + Experience Replay + Prioritized Experience Replay
Experience replay lets online reinforcement learning agents remember and reuse experiences from the past. [paper](https://arxiv.org/abs/1511.05952)

# Results

Performance is measured by the fewest number of episodes required to solve the environment.
![alt text](images/Navigation.png)

With DQN+RB I was able to solve the game in 165 episodes:
![alt text](images/results.png)

# Hyperparameters

I test different values of the 𝛆-greedy init values, starting from 0.99 to 0.985, finding the best results with 0.987.
You can find the 𝛆-greedy logic implemented as part of the Agent.act() method in dqn_agent.py of the source code.
Also different fc1_units and fc2_units was tested. 

# Future Improvements

1) Test the Rainbow Algorithm: [here](https://arxiv.org/abs/1710.02298)
