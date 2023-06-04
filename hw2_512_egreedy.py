# RACETRACK
# EXERCISE 5.12


import numpy as np
import matplotlib.pyplot as plt
import random

track2 = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

track1 = [[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3],
          [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3],
          [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3],
          [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0]]


'''Track is from virtual environment called gym that is an opensource resource on github.
But below is a modified version of the version in github'''


class Track:
    def __init__(self, track_type):
        """
        :param track_type: the type of the track to load
        """
        # All possible state type
        self.STATE_OUT = 0
        self.STATE_ON = 1
        self.STATE_START = 2
        self.STATE_END = 3

        # Maze grid
        self.track = []

        # Maze height
        self.TRACK_HEIGHT = 0

        # Maze width
        self.TRACK_WIDTH = 0

        # Max and min values of velocity in both directions
        self.MAX_VEL, self.MIN_VEL = 5, -5

        # Max absolute value of acceleration in both directions
        self.MAX_ACC = 1

        # All possible actions
        self.actions = [[a_i, a_j] for a_j in range(-self.MAX_ACC, self.MAX_ACC + 1) for a_i in
                        range(-self.MAX_ACC, self.MAX_ACC + 1)]

        # Probability that action has no effect due to failure
        self.FAILURE_PROB = 0.1

        # Maze states colors
        # self.RGB_BROWN = (139 / 255, 69 / 255, 19 / 255)
        self.RGB_BROWN = (51,51,51)
        # self.RGB_GREEN = (.5, 1, 0)
        self.RGB_GREEN = (193,255,193)
        # self.RGB_RED = (1, 0, 0)
        self.RGB_RED = (255,48,48)
        # self.RGB_YELLOW = (1, 1, 0)
        self.RGB_YELLOW = (0,100,0)
        # self.RGB_BLACK = (0, 0, 0)
        self.RGB_BLACK = (255,105,180)

        self.track = track_type
        self.set_track_ht_wd()

    def take_action(self, state, action, is_example):
        # return [new state, reward, done]
        # Current state coordinates
        i, j, v_x, v_y = state
        a_i, a_j = action

        if np.random.binomial(1, self.FAILURE_PROB) == 1 and not is_example:
            a_i, a_j = 0, 0

        p_i = i
        p_j = j
        v_x += a_i
        v_y += a_j
        i -= v_x
        j += v_y

        done = False

        states_visited = [[i_, j_] for i_ in range(min(i, p_i), max(i, p_i) + 1) for j_ in range(min(j, p_j), max(j, p_j) + 1)]
        states_visited_type = [self.track[i_][j_] if 0 <= i_ < self.TRACK_HEIGHT
                                                     and 0 <= j_ < self.TRACK_WIDTH else self.STATE_OUT for i_, j_ in states_visited]

        if self.STATE_END in states_visited_type:
            done = True
        elif self.STATE_OUT in states_visited_type:
            i, j, v_x, v_y = random.choice([[i, j, 0, 0] for i, j in self.get_state_locs(self.STATE_START)])

        return [i, j, v_x, v_y], -1, done

    def A(self, state):
        actions = []
        # All possible actions
        A = self.actions.copy()
        _, _, v_x, v_y = state

        # Discard actions that would make the speed of car negative or higher than max in at least one direction, or
        # zero in both directions
        for a in A:
            a_i, a_j = a
            if v_x + a_i < self.MIN_VEL or v_x + a_i > self.MAX_VEL:
                continue
            if v_y + a_j < self.MIN_VEL or v_y + a_j > self.MAX_VEL:
                continue
            if v_x + a_i == 0 and v_y + a_j == 0:
                continue
            actions.append(a)
        return actions

    def set_track_ht_wd(self):
        # Recompute track height and width
        self.TRACK_HEIGHT = len(self.track)
        self.TRACK_WIDTH = len(self.track[0])

    '''
    def load_track_from_csv(self, file_name):
        #Load the track from a csv file
        #:param file_name: name of the file
        
        # Load the track
        with open(file_name) as csv_file:
            self.track = [list(map(int, rec)) for rec in csv.reader(csv_file, delimiter=',')]

        # Recompute track height and width
        self.TRACK_HEIGHT = len(self.track)
        self.TRACK_WIDTH = len(self.track[0])
        '''

    def get_state_locs(self, state_type):
        """
        :param state_type: state type
        :return: a list containing all states of the track of the given type
        """
        return [[i, j] for i in np.arange(self.TRACK_HEIGHT) for j in np.arange(self.TRACK_WIDTH) if
                self.track[i][j] == state_type]

    def print_track_plot(self, state=None):
        tr_rgb = self.track.copy()

        tr_rgb = [[self.RGB_GREEN if s == self.STATE_OUT else self.RGB_BROWN if s == self.STATE_ON else
        self.RGB_YELLOW if s == self.STATE_START else self.RGB_RED for s in row] for row in tr_rgb]

        copy = tr_rgb.copy()
        if state is not None:
            for a in range(len(state)):
                x, y, _, _ = state[a]
                copy[x][y] = self.RGB_BLACK
                copy = copy.copy()
        plt.imshow(copy, origin='lower', interpolation='none')
        plt.gca().invert_yaxis()
        plt.show()


# A wrapper class for parameters of the algorithm
class Params:
    def __init__(self):
        # Discount
        self.gamma = 1
        self.episodes = 5_000

        # Number of examples to show at the end
        self.examples = 5
        self.start_q_value = -5_000

        # Soft policy b type
        self.b_policy_type = 'epsilon_greedy'
        self.epsilon = 0.1


class OffPolicyMonteCarlo:
    def __init__(self, track, params):
        self.track = track
        self.params = params

        # Shape of state-action and state space
        s_a_shape = (
            self.track.TRACK_HEIGHT, self.track.TRACK_WIDTH, self.track.MAX_VEL - self.track.MIN_VEL + 1,
            self.track.MAX_VEL - self.track.MIN_VEL + 1, 2 * self.track.MAX_ACC + 1, 2 * self.track.MAX_ACC + 1)
        ss_shape = (
            self.track.TRACK_HEIGHT, self.track.TRACK_WIDTH, self.track.MAX_VEL - self.track.MIN_VEL + 1,
            self.track.MAX_VEL - self.track.MIN_VEL + 1)

        # Initial state action pair and C values
        self.Q = np.full(s_a_shape, params.start_q_value)
        self.C = np.zeros(s_a_shape)

        # Initial policy, for each state choose a random action from ones allowed in this state
        self.pi = np.empty(ss_shape, dtype=object)
        for i in range(ss_shape[0]):
            for j in range(ss_shape[1]):
                for v_x in range(self.track.MIN_VEL, self.track.MAX_VEL + 1):
                    for v_y in range(self.track.MIN_VEL, self.track.MAX_VEL + 1):
                        self.pi[i, j, v_x, v_y] = random.choice(track.A([i, j, v_x, v_y]))

    def fix_track(self):
        i = 0
        States_list = []
        while True:
            b = self.pi
            # Generate an episode using soft policy b
            States, Actions, Rewards = self.episode(b, self.params.b_policy_type)
            G = 0
            W = 1
            # print('Episode n:', i, '\t Step needed: ', len(States))
            States_list.append(-1*len(States))
            for t in range(len(States) - 1, -1, -1):
                G = self.params.gamma * G + Rewards[t]
                self.C[tuple(States[t] + Actions[t])] += W
                self.Q[tuple(States[t] + Actions[t])] += (W / self.C[tuple(States[t] + Actions[t])]) * (G - self.Q[tuple(States[t] + Actions[t])])
                self.pi[tuple(States[t])] = random.choice([a for a in self.track.A(States[t]) if self.Q[tuple(States[t] + a)] ==
                                                       np.max([self.Q[tuple(States[t] + a)] for a in self.track.A(States[t])])])
                if Actions[t] != self.pi[tuple(States[t])]:
                    break
                W *= 1 / (1 - self.params.epsilon + self.params.epsilon/len(self.track.A(States[t])))
            i += 1
            if i > self.params.episodes:
                break
        return States_list

    def episode(self, pi, policy_type, is_example=False, S_0=None):
        # return: [states visited, action taken, reward observed]
        assert S_0 is not None or not is_example
        States = []
        Actions = []
        Rewards = []

        # Set the initial state
        if not is_example:
            States.append(random.choice([[i, j, 0, 0] for i, j in self.track.get_state_locs(self.track.STATE_START)]))
        else:
            States.append(S_0)

        idx = 0
        while True:
            if policy_type == 'deterministic':
                # Greedy
                Actions.append(pi[tuple(States[idx])])
            # elif np.random.binomial(1, self.params.epsilon) == 1 or policy_type == 'random':
            elif np.random.binomial(1, self.params.epsilon) == 1:
                # Random
                Actions.append(random.choice(self.track.A(States[idx])))
            else:
                # Greedy
                Actions.append(pi[tuple(States[idx])])
            state, reward, done = self.track.take_action(States[idx], Actions[idx], is_example)

            Rewards.append(reward)
            if done:
                break
            else:
                States.append(state)
            idx += 1
        return States, Actions, Rewards

    def generate_examples(self):
        starts = [[i, j, 0, 0] for i, j in self.track.get_state_locs(self.track.STATE_START)]
        random.shuffle(starts)
        for idx, S_0 in enumerate(starts[:self.params.examples]):
            # Make an episode
            States, Actions, Rewards = self.episode(self.pi, 'deterministic', is_example=True, S_0=S_0)
            self.track.print_track_plot(States)


def exercise5_12():
    print('Exercise 5.12')

    # Choose track type ['track1', 'track2']
    track = Track(track1)
    params = Params()
    off_policy_monte_carlo = OffPolicyMonteCarlo(track, params)
    States_list = off_policy_monte_carlo.fix_track()

    # plots
    plt.plot(States_list)
    plt.title('Plot of Rewards for each Episode')
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.show()
    plt.close()

    x_vals = list(range(4000, 5_000))
    plt.plot(x_vals, States_list[4000:5_000])
    plt.title('Plot of Rewards for each Episode')
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.show()
    plt.close()

    # Examples of trajectories following the optimal policy
    off_policy_monte_carlo.generate_examples()


def main():
    exercise5_12()


if __name__ == "__main__":
    main()
