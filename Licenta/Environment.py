import numpy as np

from Trader import Trader
from config import STATE_DIM
import config as cfg


class Environment:
    def __init__(self):
        self.trader = Trader()
        self.action_space = [0, 1]
        self.steps_till_done = 0
        self.steps_todo_done = 20

    def random_action(self):
        return self.action_space[np.random.randint(2)]

    def get_state(self):
        return self.trader.get_history_data(STATE_DIM)

    def step(self, action):
        self.steps_till_done += 1
        current_price = self.trader.get_history_data(1)[0]
        done = 0
        if self.steps_till_done % self.steps_todo_done == 0:
            done = 1
            self.steps_till_done = 0
        if action == 0:
            reward = self.trader.buy_bracket_and_return_sold(cfg.quantity, current_price)
        else:
            reward = self.trader.sell_bracket_and_return_sold(cfg.quantity, current_price)

        next_state = self.trader.get_history_data(STATE_DIM)
        return next_state, reward, done
