from policies import base_policy as bp
import numpy as np

EPSILON = 0.1
DATA_REPR_LEN = 56
DISCOUNT_FACTOR = 0.4
LEARNING_RATE = 0.15
MAX_RADIUS = 10


class Linear(bp.Policy):
    """
    A linear policy for learning the Q-function.
    """

    def cast_string_args(self, policy_args):
        policy_args['epsilon'] = float(policy_args['epsilon']) if 'epsilon' in policy_args else EPSILON
        policy_args['learning_rate'] = float(policy_args['learning_rate']) if \
            'learning_rate' in policy_args else LEARNING_RATE
        policy_args['discount_rate'] = float(policy_args['discount_rate']) if 'discount_rate' in policy_args \
            else DISCOUNT_FACTOR

        return policy_args

    def init_run(self):
        self.r_sum = 0
        self.epsilon = EPSILON
        self.discount_factor = DISCOUNT_FACTOR
        self.learning_rate = LEARNING_RATE
        self.weights = np.random.randn(DATA_REPR_LEN)
        self.max_radius = MAX_RADIUS

    def get_state_action_repr(self, state, action):
        """
        Returns a vector representation of the board data relevant for the learning process.
        for the given state and action.
        :param state: State object, represents the current status on the board.
        :param action: the next action. one of ACTIONS defined in base class.
        :return: a vector representation of the board data relevant for the learning process.
        """
        area_reper = self.get_one_hot_objects(state, action)
        dist_repr = self.get_object_min_pos_vector(state, action)
        repr = np.append(area_reper, dist_repr)
        repr = np.append(repr, np.array([1]))
        return repr

    def get_next_position(self, state, action):
        """
        Returns the coordinates of the snake's head, given that we are in a given state
        and performed the given action.
        :param state: State object, represents the current status on the board.
        :param action: the next action. one of ACTIONS defined in base class.
        :return: row and col of the next position of the snakes head in integers
        """
        head_pos, head_dir = state[1]
        next_position = head_pos.move(self.TURNS[head_dir][action])

        row = next_position[0]
        col = next_position[1]

        return row, col


    def get_object_min_pos_vector(self, state, action):
        """
        Creates a lookup table of board item and its distance according head position
        after taking given action from given state
        :param state:
        :param action:
        :return: a
        """
        min_pos_vec = np.full(11, self.max_radius)

        board = state[0]
        prev_pos, head_dir = state[1]
        board_size = prev_pos.board_size

        next_pos = self.get_next_position(state, action)
        # curr_dir = self.TURNS[head_dir][action]
        # neighbours = self.get_pos_neighbors([prev_pos], board_size, direction=curr_dir)

        # for i, pos in enumerate(neighbours):
        curr_distance = 1
        board_iter = self.pos_by_distance_iter(next_pos, prev_pos, board_size, self.max_radius)

        while curr_distance < self.max_radius:
            curr_points_batch = next(board_iter)
            if curr_points_batch is None:
                break

            for point in curr_points_batch:
                point_val = board[point[0], point[1]]
                if min_pos_vec[point_val] == self.max_radius:
                    min_pos_vec[point_val] = curr_distance

            curr_distance += 1

        # assert np.inf not in min_pos_vec
        res = np.array(min_pos_vec).flatten()
        return 1 / res

    def get_one_hot_objects(self, state, action):
        board = state[0]
        pos, head_dir = state[1]
        board_size = pos.board_size
        curr_pos = pos.pos

        area_repr = np.zeros((4, 11))
        true_dir = self.TURNS[head_dir][action]
        neighbors = self.get_pos_neighbors([curr_pos], board_size, direction=true_dir)

        for i, pos in enumerate([curr_pos] + neighbors):
            area_repr[i][board[pos[0], pos[1]]] = 1
        return area_repr.flatten()

    def get_pos_neighbors(self, curr_positions, board_size, direction=None):
        new_positions = []
        for pos in curr_positions:
            x, y = pos
            r = (x, (y + 1) % board_size[1])
            l = (x, (y - 1) % board_size[1])
            u = ((x - 1) % board_size[0], y)
            d = ((x + 1) % board_size[0], y)
            if direction is None:
                new_positions += [l, u, r, d]

            elif direction == 'N':
                new_positions += [l, u, r]

            elif direction == 'E':
                new_positions += [u, r, d]

            elif direction == 'S':
                new_positions += [r, d, l]

            elif direction == 'W':
                new_positions += [d, l, u]

        return list(set(new_positions))

    def pos_by_distance_iter(self, curr_pos, prev_pos, board_size, limit=20):
        checked_positions = {prev_pos}
        curr_positions = [curr_pos]

        for i in range(limit):
            yield curr_positions
            checked_positions.union(curr_positions)
            new_positions = set(self.get_pos_neighbors(curr_positions, board_size))  # discard visited locations
            curr_positions = new_positions.difference(checked_positions)
            curr_positions = list(curr_positions)
        yield None

    def calculate_best_action(self, state):
        """
        Calculates the best action for a given state.
        :param state: The given state.
        :return: the best actions found.
        """
        mx_q_val = -np.inf
        best_action = None

        for a in list(np.random.permutation(bp.Policy.ACTIONS)):
            sa_repr_vec = self.get_state_action_repr(state, a)
            curr_q_val = self.weights.dot(sa_repr_vec)

            if curr_q_val > mx_q_val or best_action is None:
                mx_q_val = curr_q_val
                best_action = a

        return best_action, mx_q_val

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):

        # update epsilon
        # last_learning_round = 4000
        # self.epsilon = -np.power(round / last_learning_round, 0.5) + 1
        #
        # _, best_new_Q_val = self.calculate_best_action(new_state)
        # state_repr = self.get_state_action_repr(prev_state, prev_action)
        #
        # # cur_a_weights = self.weights
        # cur_Q_val = np.dot(state_repr, self.weights)
        #
        # self.weights = self.weights - self.learning_rate * \
        #                np.multiply((cur_Q_val - (reward + 0.4 * best_new_Q_val)),
        #                            state_repr)

        if round > 4000:
            self.epsilon = 0
        else:
            self.epsilon = -np.power(round / 4000, 0.5) + 1

        best_future_action, future_opt_q_val = self.calculate_best_action(new_state)
        future_opt_state_repr = self.get_state_action_repr(new_state, best_future_action)
        # future_opt_q_val = self.weights.dot(future_opt_state_repr)

        post_action_state_repr = self.get_state_action_repr(prev_state, prev_action)
        post_action_q_val = self.weights.dot(post_action_state_repr)
        #prediction_error = reward + self.discount_factor * future_opt_q_val - post_action_q_val
        self.weights = self.weights - self.learning_rate * \
                       np.multiply((post_action_q_val - (reward + self.discount_factor * future_opt_q_val)),
                                   post_action_state_repr)
        # print(post_action_q_val)
        #self.weights -= self.learning_rate * prediction_error * post_action_state_repr

        try:
            if round % 100 == 0:
                if round > self.game_duration - self.score_scope:
                    self.log("Rewards in last 100 rounds which counts towards the score: " + str(self.r_sum), 'VALUE')
                else:
                    self.log("Rewards in last 100 rounds: " + str(self.r_sum), 'VALUE')
                self.r_sum = 0
            else:
                self.r_sum += reward

        except Exception as e:
            self.log("Something Went Wrong...", 'EXCEPTION')
            self.log(e, 'EXCEPTION')

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):

        board, head = new_state  # TODO: utilize this before making best choice
        head_pos, direction = head

        if np.random.rand() < self.epsilon:
            return np.random.choice(bp.Policy.ACTIONS)
        else:
            best_action, _ = self.calculate_best_action(new_state)
            return best_action
