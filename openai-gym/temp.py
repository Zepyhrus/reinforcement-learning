#%%
import gym
import time

env = gym.make('CartPole-v1')


#%%
for i_episode in range(1):
  observation = env.reset()
  actions = []
  rewards = []

  for t in range(100):
    env.render()
    print(observation)
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    actions.append(action)
    rewards.append(reward)
    if done:
      print('Episode finished after {} timesteps'.format(t+1))
      break
  
  env.close()






#%%
