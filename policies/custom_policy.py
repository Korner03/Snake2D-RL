from collections import defaultdict
import random
from policies import base_policy as bp
import numpy as np
import keras as ks
from keras import Sequential
from keras.datasets import mnist
from keras.layers import Dense, Conv2D, Flatten

MAX_RADIUS = 2

class Custom(bp.Policy):
    """
    custom policy - deep Q network.
    Representation - 56 features:
    first 11 - one-hot representation the next cell,
    next 33 - 3 one-hot representations of the next 3 optional neighbors,
    next 11 - the distances to the nearest instance of each of the objects
    last 1 - bias.
    """
    EPSILON = 0.1
    LEARNING_RATE = 0.01
    DISCOUNT_RATE = 0.4
    REPR_LENGTH = 44
    MEMORY_SIZE = 1000
    DIRECTIONS = ['N', 'S', 'E', 'W']

    BATCH_SIZE = 5
    DECAY = 0
    TARGET_LEARN_FREQ = 10
    LAYER_NUM = 2
    ACTIVATION = 'relu'
    FIRST_LAYER_SIZE = 1
    SECOND_LAYER_SIZE = 1
    BATCH_WEIGHT = 0.5



    def __init__(self, policy_args, board_size, stateq, actq, modelq, logq, id, game_duration, score_scope):
        """
        initialize the policy.
        :param policy_args: the arguments for the specific policy to be added as members.
        :param board_size: the shape of the board.
        :param stateq: the state queue for communication with the game.
        :param actq: the action queue for communication with the game.
        :param modelq: the model queue for communication with the game.
        :param logq: the log queue for communication with the game.
        :param id: the player ID in the game (used to understand the board states)
        :param game_duration: the duration of the game (to better set decaying parameters, like epsilon-greedy.
        :param score_scope: the number of rounds at the end of the game which count towards the score
        """
        super().__init__(policy_args, board_size, stateq, actq, modelq, logq, id, game_duration,
                         score_scope)


    def cast_string_args(self, policy_args):
        """
        this function casts arguments passed during policy construction to their proper types/names.
        :param policy_args: an arg -> string value map as received in command line.
        :return: A map of string -> value after casting to useful objects, these will be added as members to the policy
        """

        policy_args['epsilon'] = float(policy_args['epsilon']) if 'epsilon' in policy_args \
            else self.EPSILON
        policy_args['learning_rate'] = float(policy_args['learning_rate']) if \
            'learning_rate' in policy_args else self.LEARNING_RATE
        policy_args['discount_rate'] = float(policy_args['discount_rate']) if 'discount_rate' in policy_args \
            else self.DISCOUNT_RATE
        policy_args['decay'] = float(policy_args['decay']) if 'decay' in policy_args \
            else self.DECAY
        policy_args['target_learn_freq'] = int(policy_args['target_learn_freq']) if 'target_learn_freq' in policy_args \
            else self.TARGET_LEARN_FREQ
        policy_args['activation'] = policy_args['activation'] if 'activation' in policy_args \
            else self.ACTIVATION
        policy_args['layer_num'] = int(policy_args['layer_num']) if 'layer_num' in policy_args \
            else self.LAYER_NUM
        policy_args['batch_weight'] = float(policy_args['batch_weight']) if 'batch_weight' in policy_args \
            else self.BATCH_WEIGHT
        policy_args['first_size'] = float(policy_args['first_size']) if 'first_size' in policy_args \
            else self.FIRST_LAYER_SIZE
        policy_args['second_size'] = float(policy_args['second_size']) if 'second_size' in policy_args \
            else self.SECOND_LAYER_SIZE

        return policy_args


    def init_run(self):
        """
        this function is called right after the initialization of the agent.
        you may use it to initialize variables that are needed for your policy,
        such as Keras models and so on. you may also use this function
        to load your pickled model and set the variables accordingly, if the
        game uses a saved model and is not a training session.
        """
        self.r_sum = 0
        self.weights = np.random.randn(self.REPR_LENGTH)
        self.init_repr = np.zeros((10, 11))

        # arrays of the dists to the nearest object of each of the types
        # each array is of a quarter of the full diamond around the current location - in the 4 directions
        # self.top_left_dists = None
        # self.top_right_dists = None
        # self.bottom_right_dists = None
        # self.bottom_left_dists = None
        # self.search_radius = int(np.floor(0.5 * min(self.board_size))) int(np.floor(0.5 * min(self.board_size)))  # 0.5 - half the board

        self.model , self.target = self.build_model()
        self.replay_buffer = []
        self.last_learning_round = self.game_duration - self.score_scope
        self.max_radius = MAX_RADIUS


    def write_params_log(self):
        """
        log parameters.
        """
        self.log('discount_rate = ' + str(self.discount_rate))
        self.log('learning_rate = ' + str(self.learning_rate))
        self.log('decay = ' + str(self.decay))
        self.log('epsilon = ' + str(self.epsilon))


    def build_model(self):
        """
        build model of the network.
        :return:
        """
        decay = 0
        sgd = ks.optimizers.SGD(lr=self.learning_rate, decay=decay, momentum=0.0, nesterov=False)
        optimizer = sgd
        loss = 'mse'
        # activation = self.activation

        model = Sequential()

        model.add(Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu'))
        model.add(Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu'))
        model.add((Flatten()))
        model.add(Dense(units=self.REPR_LENGTH, activation='relu'))
        model.add(Dense(units=1, activation=None))

        # if self.layer_num == 2:
        #     model.add(Dense(units=int(self.REPR_LENGTH * self.second_size), activation=activation))
        #
        # model.add(Dense(units=self.REPR_LENGTH, activation='sigmoid'))
        #
        # model.add(Dense(units=1, activation=None))

        model.compile(optimizer, loss=loss, metrics=['accuracy'])

        target = ks.models.clone_model(model)  # target model

        # predict dummy value for runtime considerations
        # x = np.array([np.random.randn(self.REPR_LENGTH)])
        # dummy = model.predict(x)

        return model, target


    # def board_loc(self, row, col, board_size):
    #     """
    #     get board location (if cross an edge, apply modulo).
    #     """
    #     mod_row = row % board_size[0]
    #     mod_col = col % board_size[1]
    #     return [mod_row, mod_col]
    #
    #
    # def get_next_neighbor(self, row, col, direction, action, board_size):
    #     """
    #     the nearest neighbor according to snake's direction.
    #     """
    #     new_direction = self.TURNS[direction][action]
    #
    #     if new_direction == 'N':
    #         neighbor = self.board_loc(row-1, col, board_size)
    #
    #     elif new_direction == 'E':
    #         neighbor = self.board_loc(row, col+1, board_size)
    #
    #     elif new_direction == 'S':
    #         neighbor = self.board_loc(row+1, col, board_size)
    #
    #     else:  # new_direction == 'W'
    #         neighbor = self.board_loc(row, col-1, board_size)
    #
    #     return neighbor, new_direction
    #
    #
    #
    # def get_neighbors(self, row, col, direction, board_size):
    #     """
    #     get left right and front neighbors (according to the direction) of the cell at row, col.
    #     """
    #     neighbors = []
    #
    #     if direction == 'N':
    #         neighbors.append(self.board_loc(row, col-1, board_size))
    #         neighbors.append(self.board_loc(row-1, col, board_size))
    #         neighbors.append(self.board_loc(row, col+1, board_size))
    #
    #     elif direction == 'E':
    #         neighbors.append(self.board_loc(row-1, col, board_size))
    #         neighbors.append(self.board_loc(row, col+1, board_size))
    #         neighbors.append(self.board_loc(row+1, col, board_size))
    #
    #     elif direction == 'S':
    #         neighbors.append(self.board_loc(row, col+1, board_size))
    #         neighbors.append(self.board_loc(row+1, col, board_size))
    #         neighbors.append(self.board_loc(row, col-1, board_size))
    #
    #     elif direction == 'W':
    #         neighbors.append(self.board_loc(row+1, col, board_size))
    #         neighbors.append(self.board_loc(row, col-1, board_size))
    #         neighbors.append(self.board_loc(row-1, col, board_size))
    #
    #     return neighbors
    #
    #
    # def reduce4(self, state, action):
    #     """
    #     extract neighborhood (4 pixels) of a cells that are in the direction of the snake.
    #     :param state: a game state object
    #     :param action: an action (string)
    #     :return: a reduced representation of (state, action)
    #     """
    #     board = state[0]
    #     pos, direction = state[1]
    #     row, col = pos.pos
    #     board_size = pos.board_size
    #
    #     first_neighbor, new_direction = self.get_next_neighbor(row, col, direction, action, board_size)
    #     second_order_neighbors = self.get_neighbors(first_neighbor[0], first_neighbor[1], new_direction, board_size)
    #
    #     reduced_idx = [first_neighbor] + second_order_neighbors
    #
    #     # convert list of tuples to array indices
    #     reduced = board[[list(a) for a in list(zip(*reduced_idx))]]
    #
    #     return reduced
    #
    #
    # def move_along_ring_edge(self, board, dist, board_size, cur_row, cur_col, row_delta, col_delta, edge_dists):
    #     """
    #     walk along a ring's edge (either upper left, upper right, lower right or lower left) and collect the objects.
    #     """
    #     for i in range(dist):
    #         cur_obj = board[cur_row, cur_col]
    #         cur_cell = self.board_loc(cur_row+row_delta, cur_col+col_delta, board_size)
    #         cur_row, cur_col = cur_cell[0], cur_cell[1]
    #         edge_dists[cur_obj+1] = dist
    #
    #     return cur_row, cur_col
    #
    #
    # def update_edge_dists(self, state):
    #     """
    #     update distances to objects, on each of the 4 edges.
    #     """
    #     board = state[0]
    #     pos, direction = state[1]
    #     row, col = pos.pos
    #     board_size = pos.board_size
    #
    #     # arrays of the dists to the nearest object of each of the types
    #     # each array is of a quarter of the full diamond around the current location - in the 4 directions
    #     self.top_left_dists = np.full(11, self.search_radius)
    #     self.top_right_dists = np.full(11, self.search_radius)
    #     self.bottom_right_dists = np.full(11, self.search_radius)
    #     self.bottom_left_dists = np.full(11, self.search_radius)
    #
    #     for dist in range(self.search_radius, 0, -1):
    #         cur_cell = self.board_loc(row, col - dist, board_size)
    #         cur_row, cur_col = cur_cell[0], cur_cell[1]
    #
    #         # upper left edge
    #         cur_row, cur_col = self.move_along_ring_edge(board, dist, board_size, cur_row, cur_col, -1, 1,
    #                                                      self.top_left_dists)
    #         # upper right edge
    #         cur_row, cur_col = self.move_along_ring_edge(board, dist, board_size, cur_row, cur_col, 1, 1,
    #                                                      self.top_right_dists)
    #         # lower right edge
    #         cur_row, cur_col = self.move_along_ring_edge(board, dist, board_size, cur_row, cur_col, 1, -1,
    #                                                      self.bottom_right_dists)
    #         # lower left edge
    #         _, _ = self.move_along_ring_edge(board, dist, board_size, cur_row, cur_col, -1, -1, self.bottom_left_dists)
    #
    #
    # def build_dist2obj_repr(self, state, action):
    #     """
    #     build vector representation of the distances to each of the objects.
    #     """
    #     _, direction = state[1]
    #     new_direction = self.TURNS[direction][action]
    #
    #     if new_direction == 'N':
    #         dists = np.minimum(self.top_left_dists, self.top_right_dists)
    #
    #     elif new_direction == 'E':
    #         dists = np.minimum(self.top_right_dists, self.bottom_right_dists)
    #
    #     elif new_direction == 'S':
    #         dists = np.minimum(self.bottom_right_dists, self.bottom_left_dists)
    #
    #     else:  # new_direction == 'W'
    #         dists = np.minimum(self.bottom_left_dists, self.top_left_dists)
    #
    #     return 1 / dists


    # def build_repr_vec(self, state, action):
    #     """
    #     build vector representation of the board.
    #     """
    #     reduced = self.reduce4(state, action) + 1  # convert from [-1,9] to [0,10]
    #     repr = np.zeros((4, 11))
    #     repr[np.arange(4), reduced] = 1
    #     repr = repr.flatten()
    #
    #     repr = np.append(repr, np.array([self.build_dist2obj_repr(state, action)]))  # add dists to objects
    #     repr = np.append(repr, np.array([1]))  # add bias
    #
    #     return repr

    def get_pos_neighbors(self, curr_positions, board_size, direction=None):
        """
        returns the coordinates on board of the neighbors of the given pos. if direction is given
        the previous pos neighbor will not be included.
        :param curr_positions:
        :param board_size:
        :param direction:
        :return:
        """
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

    def get_one_hot_area(self, state, action):
        """
        returns a one hor representation of all possible 11 values on board for the 3 "next step"
        positions.
        :param state:
        :param action:
        :return:
        """
        board = state[0]
        pos, head_dir = state[1]
        board_size = pos.board_size
        # curr_pos = pos.pos
        next_position = pos.move(self.TURNS[head_dir][action])
        #
        area_repr = np.zeros(((self.max_radius * 2 + 1) ** 2, 11))
        # true_dir = self.TURNS[head_dir][action]
        rows = [i % board_size[0] for i in range(next_position[0] - self.max_radius, next_position[0] + self.max_radius + 1)]
        cols = [j % board_size[1] for j in range(next_position[1] - self.max_radius, next_position[1] + self.max_radius + 1)]
        area_poses = [(i, j) for i in rows for j in cols]

        for i, pos in enumerate(area_poses):
            area_repr[i][board[pos[0], pos[1]]] = 1
        return area_repr

    def build_repr_vec(self, state, action):
        """
        build vector representation of the board.
        """
        area_one_hot = self.get_one_hot_area(state, action)
        return area_one_hot[:, :, np.newaxis]


        # true_dir = self.TURNS[head_dir][action]
        # neighbors = self.get_pos_neighbors([next_pos], board_size, direction=true_dir)



        # reduced = self.reduce4(state, action) + 1  # convert from [-1,9] to [0,10]
        # repr = np.zeros((4, 11))
        # repr[np.arange(4), reduced] = 1
        # repr = repr.flatten()
        #
        # repr = np.append(repr, np.array([self.build_dist2obj_repr(state, action)]))  # add dists to objects
        # repr = np.append(repr, np.array([1]))  # add bias
        #
        # return repr



    def get_best_action(self, new_state):
        """
        get the action with the highest Q val.
        """
        # self.update_edge_dists(new_state)
        x = np.array([self.build_repr_vec(new_state, a) for a in bp.Policy.ACTIONS])
        predictions = self.target.predict(x)
        best_idx = np.argmax(predictions)
        best_a = bp.Policy.ACTIONS[best_idx]
        max_val = predictions[best_idx]
        return best_a, max_val


    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):
        """
        the function for choosing an action, given current state.
        it accepts the state-action-reward needed to learn from the previous
        move (which it can save in a data structure for future learning), and
        accepts the new state from which it needs to decide how to act.
        :param round: the round of the game.
        :param prev_state: the previous state from which the policy acted.
        :param prev_action: the previous action the policy chose.
        :param reward: the reward given by the environment following the previous action.
        :param new_state: the new state that the agent is presented with, following the previous action.
        :param too_slow: true if the game didn't get an action in time from the
                        policy for a few rounds in a row. use this to make your
                        computation time smaller (by lowering the batch size for example)...
        :return: an action (from Policy.Actions) in response to the new_state.
        """

        # update replay_buffer
        if prev_state != None:
            if len(self.replay_buffer) > self.MEMORY_SIZE:
                self.replay_buffer.pop(0)
            self.replay_buffer.append((prev_state, prev_action, reward))

        if np.random.rand() < self.epsilon:
            return np.random.choice(bp.Policy.ACTIONS)

        else:
            a, _ = self.get_best_action(new_state)

        return a


    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):
        """
        the function for learning and improving the policy. it accepts the
        state-action-reward needed to learn from the final move of the game,
        and from that (and other state-action-rewards saved previously) it
        may improve the policy.
        :param round: the round of the game.
        :param prev_state: the previous state from which the policy acted.
        :param prev_action: the previous action the policy chose.
        :param reward: the reward given by the environment following the previous action.
        :param new_state: the new state that the agent is presented with, following the previous action.
                          This is the final state of the round.
        :param too_slow: true if the game didn't get an action in time from the
                        policy for a few rounds in a row. you may use this to make your
                        computation time smaller (by lowering the batch size for example).
        """

        random_tuples = random.sample(self.replay_buffer, min(self.BATCH_SIZE, len(self.replay_buffer)))
        x = []
        estimated_y = []

        for t in random_tuples:
            s, a, r = t
            x.append(self.build_repr_vec(s, a))
            best_a, best_val = self.get_best_action(s)
            estimated_y.append(r + self.discount_rate * best_val)

        x = np.array(x)
        y = np.array(estimated_y)

        if self.BATCH_SIZE:
            history = self.model.fit(x, y, epochs=1, batch_size=len((y)), sample_weight=np.full(
                len(y), self.batch_weight), verbose=0)

        # learn current state, action
        cur_x = np.array([self.build_repr_vec(prev_state, prev_action)])
        best_a, best_val = self.get_best_action(new_state)
        cur_y = reward + self.discount_rate * best_val

        history = self.model.fit(cur_x, cur_y, epochs=1, batch_size=1, verbose=0)

        # update model weights
        if round % self.target_learn_freq == 0:
            self.target.set_weights(self.model.get_weights())

        # update epsilon
        self.epsilon = -np.power(round/self.last_learning_round, 0.2) + 1

        try:
            if round % 100 == 0:
                if round > self.game_duration - self.score_scope:
                    self.log("Rewards in last 100 rounds which counts towards the score: " +
                             str(self.r_sum), 'VALUE')
                else:
                    self.log("Rewards in last 100 rounds: " + str(self.r_sum), 'VALUE')
                self.r_sum = 0
            else:
                self.r_sum += reward

        except Exception as e:
            self.log("Something Went Wrong...", 'EXCEPTION')
            self.log(e, 'EXCEPTION')


    def get_params(self):
        """
        get parameters.
        """
        return 'discount_rate = ' + str(self.discount_rate) +\
            ', learning_rate = ' + str(self.learning_rate) +\
            ', decay = ' + str(self.decay) +\
            ', epsilon = ' + str(self.epsilon) + \
            ', target_learn_freq = ' + str(self.target_learn_freq) + \
            ', activation = ' + str(self.activation) + \
            ', layer_num = ' + str(self.layer_num) + \
            ', first_size = ' + str(self.first_size) + \
            ', second_size = ' + str(self.second_size) + \
               '\n'
