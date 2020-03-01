from policies import base_policy as bp
import numpy as np
import random
from keras.optimizers import SGD
from keras.models import clone_model
from keras import Sequential
from keras.layers import Dense, Conv2D, Flatten

MAX_RADIUS = 2
EPSILON = 0.3
LEARNING_RATE = 0.01
DISCOUNT_RATE = 0.5
MEMORY_SIZE = 100

BATCH_SIZE = 5
DECAY = 0
PRED_MODEL_UPDATE_FREQ = 10
BATCH_WEIGHT = 0.5

class Custom(bp.Policy):

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
            else EPSILON
        policy_args['learning_rate'] = float(policy_args['learning_rate']) if \
            'learning_rate' in policy_args else LEARNING_RATE
        policy_args['discount_rate'] = float(policy_args['discount_rate']) if 'discount_rate' in policy_args \
            else DISCOUNT_RATE
        policy_args['decay'] = float(policy_args['decay']) if 'decay' in policy_args \
            else DECAY
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

        self.train_model , self.pred_model = self.build_conv_model()
        self.replay_memory = []
        self.last_learning_round = self.game_duration - self.score_scope
        self.max_radius = MAX_RADIUS


    def build_conv_model(self):
        """
        build a convolutional neural network for q-learning.
        :return:
        """
        decay = 0
        optimizer = SGD(lr=self.learning_rate, decay=decay, momentum=0.0, nesterov=False)

        train_model = Sequential()

        train_model.add(Conv2D(filters=16, kernel_size=3, strides=(2, 2), activation='relu'))
        train_model.add((Flatten()))
        train_model.add(Dense(units=1, activation=None))

        train_model.compile(optimizer, loss='mse', metrics=['accuracy'])

        pred_model = clone_model(train_model)  # prediction model used as q-function

        return train_model, pred_model

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
        next_position = pos.move(self.TURNS[head_dir][action])

        area_repr = np.zeros(((self.max_radius * 2 + 1) ** 2, 11))
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

    def get_replay_memory_tuples(self, size):
        """
        Returns a random sample of the replay memory cached.
        :param: size - size of sample. if size larger than current cache size than return cache size sample.
        :return: a random sample of the replay memory cached.
        """
        len_random_mem = min(size, len(self.replay_memory))
        random_mem = random.sample(self.replay_memory, len_random_mem)
        reper_vec = []
        q_val_vec = []

        for mem_tup in random_mem:
            state, action, reward = mem_tup
            reper_vec.append(self.build_repr_vec(state, action))
            _, best_val = self.calculate_best_action(state)
            q_val_vec.append(reward + self.discount_rate * best_val)

        return np.array(reper_vec), np.array(q_val_vec)


    def calculate_best_action(self, state):
        """
        Calculates the best action for a given state.
        :param state: The given state.
        :return: the best actions found.
        """
        x = np.array([self.build_repr_vec(state, action) for action in bp.Policy.ACTIONS])
        predictions = self.pred_model.predict(x)
        best_idx = np.argmax(predictions)
        best_a = bp.Policy.ACTIONS[best_idx]
        max_val = predictions[best_idx]
        return best_a, max_val


    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):
        """
        Choose the best action based on the Q-function output. make a random move with probability epsilon.
        Store the previous (state, action, reward) tuple in the replay memory cache.
        :param round: round number
        :param prev_state: the previous state object
        :param prev_action: the precious action.
        :param reward: the previous reward.
        :param new_state: the new state repr object.
        :param too_slow:
        :return:
        """
        # update replay memory
        if prev_state != None:
            if len(self.replay_memory) > MEMORY_SIZE:
                self.replay_memory.pop(0)
            self.replay_memory.append((prev_state, prev_action, reward))

        # act
        if np.random.rand() < self.epsilon:
            return np.random.choice(bp.Policy.ACTIONS)

        else:
            action, _ = self.calculate_best_action(new_state)

        return action


    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):
        """
        Update the train model and pred model using the replay memory and the current state, action
        in order to improve the q-function. The learn model is update every round while the pred model which
        is the model which is used as the q function, is updated every PRED_MODEL_UPDATE_FREQ rounds.
        :param round: round number
        :param prev_state: the previous state object
        :param prev_action: the precious action.
        :param reward: the previous reward.
        :param new_state: the new state repr object.
        :param too_slow:
        """
        # use replay memory for learning
        repr_vec, q_val_vec = self.get_replay_memory_tuples(BATCH_SIZE)
        self.train_model.fit(repr_vec, q_val_vec, epochs=1, batch_size=len(q_val_vec),
                             sample_weight=np.full(len(q_val_vec), BATCH_WEIGHT), verbose=0)

        # use current state, action for learning
        curr_repr_vec = np.array([self.build_repr_vec(prev_state, prev_action)])
        _, best_val = self.calculate_best_action(new_state)
        curr_q_val = reward + DISCOUNT_RATE * best_val
        self.train_model.fit(curr_repr_vec, curr_q_val, epochs=1, batch_size=1, verbose=0)

        # update prediction model once every PRED_MODEL_UPDATE_FREQ rounds to the learn model weights.
        if round % PRED_MODEL_UPDATE_FREQ == 0:
            self.pred_model.set_weights(self.train_model.get_weights())

        # update epsilon
        self.epsilon = -np.power(round/self.last_learning_round, 0.1) + 1

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