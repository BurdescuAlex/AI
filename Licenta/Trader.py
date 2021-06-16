from time import sleep
import alpaca_trade_api as alpaca
import config as cfg


def truncate(val, decimal_places):
    return int(val * 10 ** decimal_places) / 10 ** decimal_places


class Trader(object):
    def __init__(self):
        self.key_id = cfg.ALPACA_API_KEY_ID
        self.secret_key = cfg.ALPACA_API_SECRET_KEY
        self.base_url = cfg.ALPACA_API_BASE_URL
        self.data_url = 'https://data.alpaca.markets'

        self.symbol = cfg.symbol

        self.base_bet = cfg.quantity

        # The connection to the Alpaca API
        self.api = alpaca.REST(
            self.key_id,
            self.secret_key,
            self.base_url
        )

        try:
            self.position = int(self.api.get_position(self.symbol).qty)
        except Exception as e:
            # No position exists
            self.position = 0

        # Figure out how much money we have to work with, accounting for margin
        account_info = self.api.get_account()
        self.equity = float(account_info.equity)
        self.margin_multiplier = float(account_info.multiplier)

        print(f'Account status is {account_info.status}')
        print(f'Balance = {self.equity}')
        self.market_status()
        self.check_if_tradable()

    def sell(self, quantity, symbol_price):
        succeded = False

        while not succeded:
            succeded = True
            try:
                self.current_order = self.api.submit_order(
                    symbol=cfg.symbol,
                    qty=quantity,
                    side='sell',
                    type='market',
                    time_in_force='day',
                    order_class='bracket',
                    stop_loss=dict(stop_price=f'{symbol_price * cfg.take_profit_percent}'),
                    take_profit=dict(limit_price=f'{symbol_price * cfg.stop_loss_percent}'),
                )
            except:
                sleep(0.5)
                symbol_price = self.get_history_data(1)[0]
                succeded = False

        print(
            f'Selling {quantity} at {symbol_price} with stop_loss {symbol_price * cfg.take_profit_percent} and take_profit {symbol_price * cfg.stop_loss_percent}')

    def buy(self, quantity, symbol_price):
        succeded = False

        while not succeded:
            succeded = True
            try:
                self.current_order = self.api.submit_order(
                    symbol=cfg.symbol,
                    qty=quantity,
                    side='buy',
                    type='market',
                    time_in_force='day',
                    order_class='bracket',
                    stop_loss=dict(stop_price=f'{symbol_price * cfg.stop_loss_percent}'),
                    take_profit=dict(limit_price=f'{symbol_price * cfg.take_profit_percent}'),
                )
            except:
                sleep(0.5)
                symbol_price = self.get_history_data(1)[0]
                succeded = False

        print(
            f'Buying {quantity} at {symbol_price} with stop_loss {symbol_price * cfg.stop_loss_percent} and take_profit {symbol_price * cfg.take_profit_percent}')

    def buy_bracket_and_return_sold(self, quantity, symbol_price):
        self.buy(quantity, symbol_price)

        sleep(2)

        left_bracket_id = self.current_order.legs[0].id
        right_bracket_id = self.current_order.legs[1].id

        while True:
            left_bracket_order = self.api.get_order(left_bracket_id)
            right_bracket_order = self.api.get_order(right_bracket_id)

            if left_bracket_order.status == 'filled':
                return 1  # float(left_bracket_order.limit_price) - symbol_price  # left_bracket_order reward
            elif right_bracket_order.status == 'filled':
                return -1  # - (symbol_price - float(right_bracket_order.stop_price))  # right_bracket_order reward
            elif left_bracket_order.status == 'cancelled' and right_bracket_order.status == 'cancelled':
                return 0

            sleep(0.5)

    def sell_bracket_and_return_sold(self, quantity, symbol_price):
        self.sell(quantity, symbol_price)

        sleep(2)
        left_bracket_id = self.current_order.legs[0].id
        right_bracket_id = self.current_order.legs[1].id

        while True:
            left_bracket_order = self.api.get_order(left_bracket_id)
            right_bracket_order = self.api.get_order(right_bracket_id)

            if left_bracket_order.status == 'filled':
                return 1  # - (float(left_bracket_order.limit_price) - symbol_price)  # left_bracket_order reward
            elif right_bracket_order.status == 'filled':
                return -1  # symbol_price - float(right_bracket_order.stop_price)  # right_bracket_order reward
            elif left_bracket_order.status == 'cancelled' and right_bracket_order.status == 'cancelled':
                return 0

            sleep(0.5)

    def get_history_data(self, number):
        prices = self.api.get_barset(cfg.symbol, 'minute', limit=number)
        prices = prices[cfg.symbol]
        return [price.c for price in prices]

    def check_if_tradable(self):
        try:
            asset = self.api.get_asset(self.symbol)

            if asset.tradable:
                print(f'We can trade {self.symbol}')
                return True

            print(f'We can not trade {self.symbol}')
            return False
        except Exception as e:
            print(f'We can not trade {self.symbol}')
            print(e)

            return False

    def market_status(self):
        clock = self.api.get_clock()
        print(f"The market is {'open.' if clock.is_open else 'closed.'}")
        return clock.is_open
