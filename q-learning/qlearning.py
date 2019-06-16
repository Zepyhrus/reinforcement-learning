import numpy as np
import pandas as pd
import time

N_STATES = 6
ACTIONS = ['L', 'R']
EPSILON = 0.9
ALPHA = 0.1
GAMMA = 0.9

MAX_EPISODES = 13
FRESH_TIME = 0.3

def build_q_table(n_states, actions):
  table = pd.DataFrame(
    np.zeros((n_states, len(actions))),  # initialize q_table with all 0
    columns=actions  # action names
  )

  return table
