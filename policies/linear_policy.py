from policies import base_policy as bp
import numpy as np

EPSILON = 0.05

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

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):

        # implemet code here... keep rest of code

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

        board, head = new_state
        head_pos, direction = head

        if np.random.rand() < self.epsilon:
            return np.random.choice(bp.Policy.ACTIONS)

        else:
            for a in list(np.random.permutation(bp.Policy.ACTIONS)):
                # implement code here
                pass

            # if all positions are bad:
            return np.random.choice(bp.Policy.ACTION)