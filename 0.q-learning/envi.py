"""
  This is the environment of the problem
"""
#%%
import pandas as pd
import numpy as np
import numpy.random as npr
import time


class Envi():
  def __init__(self, size):
    self.status = 0
    self.size = size
    # self.is_terminated = 0  # if self.status == 6, Envi is terminated

  def show(self):
    env = ['-'] * self.size + ['T']
    if self.status == self.size:
      pass
    else:
      env[self.status] = 'o'
    
    print('\r{}'.format(''.join(env)), end='')
    time.sleep(0.2)

  def move(self, action):
    assert action in ['l', 'r'], 'Wrong action'
    if action == 'l':
      self.status = max(0, self.status - 1)
    else:
      self.status = min(self.size+1, self.status+1)




epsilon = 0.9
gamma = 0.9
alpha = 0.1



#%%
# if __name__ == '__main__':
size = 6
q_table = pd.DataFrame(np.zeros((size+1, 2)), columns=['l', 'r'])

for i in range(13):
  print('\t :{}'.format(i), end='')
  envi = Envi(size)
  envi.show()

  while envi.status < size:
    s = envi.status

    if npr.uniform() > epsilon or (q_table.iloc[s, :] == 0.0).all():
      action = npr.choice(q_table.columns)
    else:
      action = q_table.iloc[s, :].idxmax()

    r = 0

    if action == 'r':
      if s == size - 1:
        r = 1
      s_ = q_table.index[s+1]
    else:
      s_ = q_table.index[max(0, s-1)]

    q_table.loc[s, action] = q_table.loc[s, action] + \
      alpha * (r + gamma * q_table.loc[s_, :].max() - q_table.loc[s, action])
    

    envi.move(action)
    envi.show()

print(q_table)



