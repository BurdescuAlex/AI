import mysql.connector
from plothelper import Plotter
from config import symbol, STATE_DIM, take_profit_percent, stop_loss_percent,quantity

dbconfig = {
    "host": "localhost",
    "database": "licenta",
    "user": "root",
    "password": ""
}
db = mysql.connector.connect(**dbconfig)
cursor = db.cursor()
cursor.execute(
    "SELECT `price`\
     FROM `pricing_data`\
     WHERE trade_id LIKE %s AND market_hours LIKE 'REGULAR_MARKET'",
    (symbol,))

myplotter = Plotter()


class DBenv:
    def __init__(self):
        self.states = []
        self.steps_till_done = 0
        self.steps_todo_done = 20
        self.start_money = 100000

    def state(self):
        if len(self.states) == 0:
            for i in range(0, STATE_DIM):
                price = cursor.fetchone()[0]
                myplotter.add_prices(price)  # plot helper
                self.states.append(price)
        else:
            price = cursor.fetchone()[0]
            self.states.append(price)
            myplotter.add_prices(price) # plot helper
            self.states.pop(0)
        return self.states

    def step(self, action):
        self.steps_till_done += 1
        current_price = self.states[0]
        done = 0
        reward = 0
        if self.steps_till_done % self.steps_todo_done == 0:
            done = 1
            self.steps_till_done = 0
        if action == 0:  # BUY
            stopwhile = False
            while not stopwhile:
                next_price = cursor.fetchone()[0]
                myplotter.add_prices(next_price)  # plot helper
                self.states.append(next_price)
                self.states.pop(0)
                if next_price >= current_price * take_profit_percent:
                    self.start_money += (-current_price + next_price) * quantity
                    reward = 1
                    stopwhile = True
                elif next_price <= current_price * stop_loss_percent:
                    reward = -1
                    stopwhile = True
                    self.start_money += (-current_price + next_price) * quantity
        else:  # SELL
            stopwhile = False
            while not stopwhile:
                next_price = cursor.fetchone()[0]
                myplotter.add_prices(next_price)  # plot helper
                self.states.append(next_price)
                self.states.pop(0)
                if next_price <= current_price * stop_loss_percent:
                    self.start_money += (current_price - next_price) * quantity
                    reward = 1
                    stopwhile = True
                elif next_price >= current_price * take_profit_percent:
                    reward = -1
                    stopwhile = True
                    self.start_money += (current_price - next_price) * quantity
        print(f'Money: {self.start_money}')
        myplotter.add_money(self.start_money)
        return self.states, reward, done
