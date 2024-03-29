"""



"""
import sys
sys.path.extend(['4.0.DQN'])

from maze_env import Maze
from DQN_brain import DeepQNetwork


def run_maze():
  step = 0
  for episode in range(1000):
    # initial observation
    observation = env.reset()

    while True:
      # fresh env
      env.render()

      # RL choose action based on observation
      action = RL.choose_action(observation)

      # RL take action and get next observation and reward
      observation_, reward, done = env.step(action)

      RL.store_transition(observation, action, reward, observation_)

      if (step > 200) and (step % 5 == 0):
        RL.learn()

      # swap observation
      observation = observation_

      # break while loop when end of this episode
      if done:
        break
      step += 1

  # end of game
  print('game over')
  env.destroy()


if __name__ == "__main__":
  # maze game
  env = Maze()
  RL = DeepQNetwork(
    n_features=env.n_features, 
    n_actions=env.n_actions, 
    learning_rate=0.01,
    reward_decay=0.9,
    e_greedy=0.9,
    replace_target_iter=300,
    memory_size=2000,
    batch_size=64,
    e_greedy_increment=None,
    output_graph=True)
  
  env.after(100, run_maze)
  env.mainloop()
  RL.show_his()