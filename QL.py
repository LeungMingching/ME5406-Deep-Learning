import numpy as np

BOARD_ROWS = 4
BOARD_COLS = 4
FRISBEE = [(3, 3)]
HOLES = [(1, 1), (1, 3), (2, 3), (3, 0)]
START = (0, 0)

EXPLORATION_RATE = 0.2
DISCOUNT_RATE = 0.9
LEARNING_RATE = 0.9
MAX_STEPS = 10000


def generate_grid(row=BOARD_ROWS, col=BOARD_COLS, start_point=None, goal=None, holes_percentage=None):
    global BOARD_ROWS, BOARD_COLS, START, FRISBEE, HOLES

    # generate win state
    if goal is None:
        goal = FRISBEE
    elif goal == 'R':
        i = np.random.choice(range(row))
        j = np.random.choice(range(col))
        goal = [(i, j)]

    # generate starting point
    if start_point is None:
        start_point = START
    elif start_point == 'R':
        i = np.random.choice(range(row))
        j = np.random.choice(range(col))
        start_point = (i, j)

    # generate lose states
    holes = []
    if holes_percentage is None:
        holes = HOLES
    else:
        holes_no = round(row * col * holes_percentage)

        n = 0
        while n < holes_no:
            i = np.random.choice(range(row))
            j = np.random.choice(range(col))

            if ((i, j) not in holes) and ((i, j) not in goal) and ((i, j) != start_point):
                holes.append((i, j))
                n += 1

    # modify the global variables
    BOARD_ROWS, BOARD_COLS, START, FRISBEE, HOLES = row, col, start_point, goal, holes

    return row, col, start_point, goal, holes


def show_grid(grid):
    row, col, start_point, goal, holes = grid

    for i in range(row):
        out = ''
        for j in range(col):
            state = (i, j)
            if state == start_point:
                token = 'S'

            elif state in goal:
                token = 'G'

            elif state in holes:
                token = 'X'

            else:
                token = '-'

            out += token + ' '
        print(out)


def check_connectivity(grid):
    # from Machine Vision
    # scan rot180 scan rot180 scan
    # CAN BE IMPROVED

    row, col, start_point, goal, holes = grid

    board = np.zeros((row, col))

    for i in range(row):
        for j in range(col):
            if (i, j) in holes:
                board[i][j] = -1

    col_in = -np.ones(row)
    row_in = -np.ones(col + 2)
    board = np.insert(board, 0, values=col_in, axis=1)
    board = np.insert(board, col + 1, values=col_in, axis=1)
    board = np.insert(board, 0, values=row_in, axis=0)
    board = np.insert(board, row + 1, values=row_in, axis=0)

    la = 1
    for i in range(1, row + 1):
        for j in range(1, col + 1):
            if board[i][j] == 0:

                left = board[i][j - 1]
                top = board[i - 1][j]
                temp_lu = [left, top]

                if temp_lu == [-1, -1]:
                    board[i][j] = la
                    la += 1
                else:
                    temp2_lu = [t for t in temp_lu if t != -1]
                    board[i][j] = min(temp2_lu)

    board = np.rot90(board, k=2)

    for i in range(1, row + 1):
        for j in range(1, col + 1):
            if board[i][j] != -1:

                left = board[i][j - 1]
                top = board[i - 1][j]
                temp_lu = [left, top]

                if temp_lu != [-1, -1]:
                    temp2_lu = [t for t in temp_lu if t != -1]
                    board[i][j] = min(temp2_lu)

    board = np.rot90(board, k=2)

    for i in range(1, row + 1):
        for j in range(1, col + 1):
            if board[i][j] != -1:

                left = board[i][j - 1]
                top = board[i - 1][j]
                temp_lu = [left, top]

                if temp_lu != [-1, -1]:
                    temp2_lu = [t for t in temp_lu if t != -1]
                    board[i][j] = min(temp2_lu)

    board = board[1:row + 1, 1:col + 1]

    if board[start_point] == board[goal[0]]:
        return True
    else:
        return False


class State:
    def __init__(self, state=START):
        self.isEnd = False
        self.state = state

    def get_reward(self, state=None):
        if state is None:
            state = self.state
        if state in FRISBEE:
            return 1
        elif state in HOLES:
            return -1
        else:
            return 0

    def fun_end(self):
        if (self.state in FRISBEE) or (self.state in HOLES):
            self.isEnd = True

    def nxt_pos(self, action):
        if action == 'up':
            nxt_state = (self.state[0] - 1, self.state[1])
        elif action == 'down':
            nxt_state = (self.state[0] + 1, self.state[1])
        elif action == 'left':
            nxt_state = (self.state[0], self.state[1] - 1)
        else:
            nxt_state = (self.state[0], self.state[1] + 1)

        if (nxt_state[0] >= 0) and (nxt_state[0] <= BOARD_ROWS - 1):
            if (nxt_state[1] >= 0) and (nxt_state[1] <= BOARD_COLS - 1):
                return nxt_state
        return self.state


class Agent:
    def __init__(self):
        self.State = State()
        self.path = []
        self.actions = ['up', 'down', 'left', 'right']
        self.step_no = 0
        self.g = 0
        self.greedy_path = []
        self.give_up = False

        # Q table initialization
        self.q_table = {}
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                self.q_table[(i, j)] = {}
                for a in self.actions:
                    self.q_table[(i, j)][a] = 0

    def pick_action(self, exp_rate=EXPLORATION_RATE):
        # default e-greedy policy with EXPLORATION RATE， input 0 to be greedy policy
        current_pos = self.State.state
        max_qsa = -2
        action = []

        if np.random.uniform(0, 1) <= exp_rate:
            action = [np.random.choice(self.actions)]
        else:
            for a in self.actions:
                if self.q_table[current_pos][a] > max_qsa:
                    action = [a]
                    max_qsa = self.q_table[current_pos][a]
                elif self.q_table[current_pos][a] == max_qsa:
                    action.append(a)
        action = np.random.choice(action)
        return action

    def take_action(self, action):
        pos = self.State.nxt_pos(action)
        return State(state=pos)  # once take action, update the State class

    def find_max_qa(self, state):
        max_qsa = -2.0
        for a in self.q_table[state]:
            if self.q_table[state][a] > max_qsa:
                max_qsa = self.q_table[state][a]
        return max_qsa

    def reset(self):
        self.path = []
        self.State = State(START)
        self.step_no = 0

    def show_result(self):

        path_states = [s[0] for s in self.greedy_path]

        for i in range(BOARD_ROWS):
            out = ''
            for j in range(BOARD_COLS):
                state = (i, j)

                if state in path_states:
                    ind = path_states.index(state)
                    a = self.greedy_path[ind][1]
                    if a == 'up':
                        token = '↑ '
                    elif a == 'down':
                        token = '↓ '
                    elif a == 'left':
                        token = '← '
                    else:
                        token = '→ '

                elif state in FRISBEE:
                    token = 'G '

                elif state in HOLES:
                    token = 'X '

                else:
                    token = '- '

                out += token
            print(out)

    def q_learning(self, rounds=10):
        i = 0.0

        # loop for episode
        while i < rounds:

            # loop for steps
            while not self.State.isEnd and self.step_no < MAX_STEPS:

                # pick action with e-greedy policy
                action = self.pick_action()

                # counting steps
                self.step_no += 1

                # record the path
                self.path.append([self.step_no, self.State.state, action])

                # update State
                self.State = self.take_action(action)

                # read reward
                reward = self.State.get_reward()

                # read S & A from the recorded path
                last_state = self.path[-1][1]
                last_action = self.path[-1][2]

                # find the max q(s'a')
                max_qsa = self.find_max_qa(self.State.state)

                # update q_table
                self.q_table[last_state][last_action] = \
                    self.q_table[last_state][last_action] + \
                    LEARNING_RATE * (reward + DISCOUNT_RATE * max_qsa - self.q_table[last_state][last_action])

                # determine if game is end
                self.State.fun_end()

                # show win or lose
                if self.State.isEnd:
                    if reward > 0:
                        print('Win:', reward)
                    elif reward < 0:
                        print('Lose:', reward)
                    else:
                        print('End with reward = o. Error!!')

            if self.step_no < MAX_STEPS:  # only ending up to an end state can be considered as a valid episode
                i += 1
            self.reset()

        # final play with greedy policy (CAN BE IMPROVED)
        while not self.State.isEnd:

            # pick action with greedy policy
            action = self.pick_action(0)

            # counting steps in 1 episode
            self.step_no += 1

            # record the greedy path
            self.greedy_path.append([self.State.state, action])

            # update StateFRISBEE
            self.State = self.take_action(action)

            # determine if game is end
            self.State.fun_end()

            if self.step_no > MAX_STEPS:
                self.give_up = True
                print('current optimal path cannot converge to an end. Please increase looping time...')
                break


# ag = Agent()
# ag.q_learning(1000)
# if not ag.give_up:
#     print(ag.q_table)
#     print('optimal path:', ag.greedy_path)
#     ag.show_result()
#
#
# grid_world = generate_grid(10, 10, 'R', 'R', 0.25)
# print(grid_world)
# show_grid(grid_world)
# print(check_connectivity(grid_world))

connected = False
while not connected:
    grid_world = generate_grid(10, 10, 'R', 'R', 0.25)  # row, col, start_point, goal[()], holes_percentage
    connected = check_connectivity(grid_world)
show_grid(grid_world)

ag = Agent()
ag.q_learning(5000)
if not ag.give_up:
    # print(ag.q_table)
    print('optimal path:', ag.greedy_path)
    ag.show_result()
