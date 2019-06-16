# Reinforcement Learning (RL)
A study environment of reinforcement learning, including any implementation of RL algorithm and papers


# Different types of RL algorithms
## Model
1. Model-free RL: Do not try to understand the environment;
2. Model-based RL:
    * Try to understand (model) the environment;
    * Create a envi (environment) simulator, to simulate all possible actions;

## Actions
1. Policy-based: 
    * Policy Gradients;
2. Value-based:
    * Q learning;
    * Sarsa;
3. Actor-Critic:
    * Combined the benefits of policy-based and value-based;
    * Accelerate learning speed of Policy Gradient;

## Update
1. Round:
    * Monte-carlo learning;
    * Basic policy gradients
2. Step:
    * Q leaning;
    * Sarsa;
    * Upgraded policy gradients;
It should be noted that, some of the RL probelm does not belong to any round game.

## Learning policy
1. Online learning:
    * Sarsa/Sarsa lambda;
2. Off-line learning:
    * Q learning;
    * Deep-Q-Network;


# Q Learning
Initialize *Q(s, a)* arbitrarily\n
Repeat (for each episode):\n
    Initialize *s*\n
    Repeat (for each step of episode):\n
        Choose *a* form *s* using policy derived from Q (e.g., \epsilon-greedy)\n
        Take action *a*, obeserve *r, s'*\n
        *Q(s, a) <- Q(s, a) + \alpha\[r+\gamma max_a' Q(s', a') - Q(s, a)\]*\n
        *s <- s';*\n
    until *s* in terminal; \n


# Acoknowledgement




