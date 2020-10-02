import numpy as np

BOARD_ROWS = 4
BOARD_COLS = 4
FRISBEE = [(3, 3)]
HOLES = [(1, 1), (3, 0), (1, 3), (2, 3)]
START = (0, 0)

EXPLORATION_RATE = 0.2
DISCOUNT_RATE = 0.9
MAX_STEPS = 2000


class State:
    def __init__(self, state=START):
        self.isEnd = False
        self.state = state
        self.board = np.zeros([BOARD_ROWS, BOARD_COLS])

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
        self.first_visit = False
        self.greedy_path = []
        self.give_up = False

        # Q table initialization
        self.q_table = {}
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                self.q_table[(i, j)] = {}
                for a in self.actions:
                    self.q_table[(i, j)][a] = 0

        # initialize return
        self.ret = {}
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                self.ret[(i, j)] = {}
                for a in self.actions:
                    self.ret[(i, j)][a] = [0, 0]  # [sum of g, times of first visit]

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

    def deter_first_visit(self, current_nsa, whole_path):

        self.first_visit = False

        # obtain all same (s, a)
        same_nsa = [a for a in whole_path if a[1:2] == current_nsa[1:2]]

        # determine if the first (s, a)
        if len(same_nsa) == 1 or current_nsa[0] == same_nsa[0][0]:
            self.first_visit = True

    def reset(self):
        self.path = []
        self.State = State()
        self.step_no = 0
        self.g = 0

    # def show_result(self):
    #
    #     # initialise all the states
    #     states = []
    #     for i in range(BOARD_ROWS):
    #         for j in range(BOARD_COLS):
    #             states.append((i, j))
    # 
    #     # scan the grid
    #     for s in states:
    #         if s ==

    def mc(self, rounds=10):
        i = 0.0

        while i < rounds:

            # forward moving
            while not self.State.isEnd and self.step_no < MAX_STEPS:
                # pick action with e-greedy policy
                action = self.pick_action()

                # counting steps in 1 episode
                self.step_no += 1

                # record the path in 1 episode
                self.path.append([self.step_no, self.State.state, action])

                # update State
                self.State = self.take_action(action)

                # determine if game is end
                self.State.fun_end()

            # backward calculate for q_values

            # if game ends, get reward
            reward = self.State.get_reward()
            for a in self.actions:
                self.q_table[self.State.state][a] = reward
            if reward > 0:
                print('Bingo:', reward)
            else:
                print('Fail:', reward)

            # loop for each step
            for s in reversed(self.path):

                self.g = DISCOUNT_RATE * self.g + reward
                reward = self.State.get_reward(s[1])

                # determinate if it is first-visit
                self.deter_first_visit(s, self.path)

                # update Q_table
                if self.first_visit:
                    # append g to g_list
                    self.ret[s[1]][s[2]][0] += self.g
                    self.ret[s[1]][s[2]][1] += 1

                    self.q_table[s[1]][s[2]] = self.ret[s[1]][s[2]][0] / self.ret[s[1]][s[2]][1]

            self.reset()
            i += 1

        #  final play with greedy policy (CAN BE IMPROVED)
        while not self.State.isEnd:

            # pick action with greedy policy
            action = self.pick_action(0)

            # counting steps in 1 episode
            self.step_no += 1

            # record the greedy path
            self.greedy_path.append([self.State.state, action])

            # update State
            self.State = self.take_action(action)

            # determine if game is end
            self.State.fun_end()

            if self.step_no > MAX_STEPS:
                self.give_up = True
                print('current optimal path cannot converge to an end. Please increase looping time...')
                break


if __name__ == "__main__":
    ag = Agent()
    ag.mc(2000)
    if not ag.give_up:
        print(ag.q_table)
        print('optimal path:', ag.greedy_path)

    # put all the code in final_move in a function show_result.
    # let self.give_up be the termination factor of the main while.
    # complete show_result
    # need a threshold value to terminate the whole process
