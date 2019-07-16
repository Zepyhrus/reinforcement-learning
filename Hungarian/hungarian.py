#%%
import pandas as pd
import numpy as np
import numpy.random as npr

npr.seed(98)
dim = 10

#%%
weight = np.ones((dim, dim)).astype(np.int)


# initialize the weight matrix
#   0: legal routine
#   1: illegial routine
for i in range(dim**2):
  weight[npr.randint(dim), npr.randint(dim)] = 0
#%%
# boys, as the column
# girls, as the row
#      girls
# boys  0   1   2   3   4   5   6 ...
# boys  1
# boys  2
#       .
#       .
#       .
boys = -np.ones(dim).astype(np.int)
girls = -np.ones(dim).astype(np.int)

MATCH = []
OCCUPIED = np.zeros(dim).astype(np.int)

def find(boy):
  # loop over girls
  for girl in range(dim):
    # if the boy and the girl are both attractive to each other and the girl
    #   is available:
    if weight[boy][girl] and not OCCUPIED[girl]:
      OCCUPIED[girl] = 1
      if girls[girl] < 0 or find(girls[girl]):
        girls[girl] = boy
        MATCH.append((boy, girl))
        return True
  
  return False

for boy in range(dim):
  find(boy)
#%%
