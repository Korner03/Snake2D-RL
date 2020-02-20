from policies import base_policy as bp
import numpy as np

EPSILON = 0.05
DATA_REPR_LEN = 20
DISCOUNT_FACTOR = 0.98
LEARNING_RATE = 0.05


class Linear(bp.Policy):
    """
    A linear policy for learning the Q-function.
    """
    def cast_string_args(self, policy_args):
        policy_args['epsilon'] = float(policy_args['epsilon']) if 'epsilon' in policy_args else EPSILON
        return policy_args

    def init_run(self):
        self.r_sum = 0
        self.epsilon = EPSILON
        self.discount_factor = DISCOUNT_FACTOR
        self.learning_rate = LEARNING_RATE
        self.weights = np.random.normal((DATA_REPR_LEN,))

    def get_state_action_repr(self, state, action):
        """
        Returns a vector representation of the board data relevant for the learning process.
        for the given state and action.
        :param state: State object, represents the current status on the board.
        :param action: the next action. one of ACTIONS defined in base class.
        :return: a vector representation of the board data relevant for the learning process.
        """
        pass

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
        pass

    def calculate_best_action(self, state):
        """
        Calculates the best action for a given state.
        :param state: The given state.
        :return: the best actions found.
        """
        mx_q_val = 0
        best_action = None

        for a in list(np.random.permutation(bp.Policy.ACTIONS)):
            sa_repr_vec = self.get_state_action_repr(state, a)
            curr_q_val = self.weights.dot(sa_repr_vec)

            if curr_q_val > mx_q_val or best_action is None:
                mx_q_val = curr_q_val
                best_action = a
        return best_action

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):

        # implemet code here... keep rest of code

        best_future_action = self.calculate_best_action(new_state)
        future_opt_state_repr = self.get_state_action_repr(new_state, best_future_action)
        future_opt_q_val = self.weights.dot(future_opt_state_repr)

        post_action_state_repr = self.get_state_action_repr(prev_state, prev_action)
        post_action_q_val = self.weights.dot(post_action_state_repr)

        prediction_error = reward + self.discount_factor * future_opt_q_val - post_action_q_val
        self.weights -= self.learning_rate * prediction_error * post_action_state_repr

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
            return self.calculate_best_action(new_state)
