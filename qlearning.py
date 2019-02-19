import numpy as np
import random

gamma = 0.8
episode = 1000

'''
- Matrice size: 4 X 5
- States: (0, 1, 2, 3) = (Start, state next to starting position, mine, destination), respectively
- Actions: (0, 1, 2, 3, 4) = (Up, down, left, right, do nothing)
'''
if __name__ == '__main__':
    episode_flag = True
    iteration = 0
    q_table = np.zeros((4, 5))
    '''
    Goal = +10
    Mine = -10
    A single move = -1
    '''
    reward_table = np.array([[0, -10, 0, -1, -1],
                             [0, 10, -1, 0, -1],
                             [-1, 0, 0, 10, -1],
                             [-1, 0, -10, 0, 10]])
    # -1 : invalid transition
    transition_matrix = np.array([[-1, 2, -1, 1, 1],
                                  [-1, 3, 0, -1, 2],
                                  [0, -1, -1, 3, 3],
                                  [1, -1, 2, -1, 4]])
    # Rows: states
    # Columns: valid actions
    valid_actions = np.array([[1, 3, 4],
                              [1, 2, 4],
                              [0, 3, 4],
                              [0, 2, 4]])
    prev_q_tables = []
    match_cnt = 0
    for i in range(episode):
        if not episode_flag:
            print('Stop Q-table update iteration.')
            break
        start_state = 0
        current_state = start_state
        while(current_state != 3):
            # repeat until the agent reaches the destination
            current_action = random.choice(valid_actions[current_state])
            reward = reward_table[current_state][current_action]
            next_state = transition_matrix[current_state][current_action]
            discounted_future_value = gamma * np.max(q_table[next_state])
            new_q_val = reward + discounted_future_value
            q_table[current_state][current_action] = new_q_val
            print(q_table)
            current_state = next_state
            if(current_state == 3):
                print('The agent has reached the destination!')
                iteration += 1
        # if previous N matrices have the same value, terminate (i.e. N = 10)
        if len(prev_q_tables) < 10:
            prev_q_tables.append(q_table)
        else:
            previous_tables = prev_q_tables
            for T in previous_tables:
                # Element-wise comparison between each element in the current Q-table and prev Q-tables
                if((q_table == T).all()):
                    match_cnt += 1
                    if match_cnt == len(previous_tables):
                        episode_flag = False
                        break
                else:
                    continue
            prev_q_tables.pop(0)
    print('Episode iteration: ', iteration)
    print('< Final Q-matrix >\n', q_table)
