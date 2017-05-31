from __future__ import print_function
import numpy as np
import pandas as pd
import time


EPSILON = 0.9
N_STATES = 6
ACTIONS = ["left", "right"]
FRESH_TIME = 0.3
GAMMA = 0.9
learning_rate = 0.1


def build_q_table(nb_states):
    table = pd.DataFrame(
        np.zeros((nb_states, len(ACTIONS))),     # q_table initial values
        columns=ACTIONS,    # actions's name
    )
    return table

def agent(state, q_table):
    state_actions = q_table.iloc[state,:]
    if (np.random.uniform()<EPSILON) or (state_actions.all()==0):
        action = np.random.choice(ACTIONS)
    else:
        action = np.argmax(state_actions)

    return action

def simulator(state, action):
    if action == "right":
        if state == N_STATES - 2:   # terminate
            state_next = 'terminal'
            reward = 1
        else:
            state_next = state + 1
            reward = 0
    else:
        reward = 0
        if state == 0:
            state_next = state
        else:
            state_next = state - 1
    return state_next, reward

def update_env(S, episode, step_counter):
    # This is how environment be updated
    env_list = ['-']*(N_STATES-1) + ['T']   # '---------T' our environment
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print ('\r{}'.format(interaction), end = "")
        time.sleep(2)
        print('\r', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)

# main
nb_max_iterations = 14



q_table = build_q_table(N_STATES)

for i in range(nb_max_iterations):
    step_counter = 0
    state = 0
    flag_terminal = False
    update_env(state, i, step_counter)
    while not flag_terminal:
        action = agent(state,q_table)
        state_next, reward = simulator(state, action)
        q_prediction = q_table.ix[state,action]
        if flag_terminal != True:
            print (q_table.iloc[state_next, :].max())
            q_target = reward + GAMMA * q_table.iloc[state_next, :].max()
        else:
            q_target = reward
            flag_terminal = True

        q_table.ix[state, action] += learning_rate * (q_target - q_prediction)
        state = state_next  # move to next state

        update_env(state, i, step_counter + 1)
        step_counter += 1


print (q_table)



