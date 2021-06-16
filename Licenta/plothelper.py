import matplotlib.pyplot as plt


class Plotter:
    def __init__(self):
        self.money = []
        self.prices = []

    def add_money(self, amount):
        self.money.append(float(amount))

    def add_prices(self, price):
        self.prices.append(float(price))

    def plotgraph(self):
        plt.plot(self.money)
        plt.title('Evolution of Money')
        plt.show()
        plt.plot(self.prices)
        plt.title('Evolution of Stock Price')
        plt.show()
