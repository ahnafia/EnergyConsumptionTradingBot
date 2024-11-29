import pandas as pd
from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from datetime import datetime

API_KEY = "PKLM2IXH3UVHJHWLTJ2B"
API_SECRECT = "EIzrfbFBdVaeZI8QtorItG1AIu9IBYM6HaLZfdqe"
BASE_URL = "https://paper-api.alpaca.markets/v2"

ALPACA_CREDS = {
    "API_KEY": API_KEY,
    "API_SECRET": API_SECRECT,
    "PAPER": True,
    "BASE_URL" : BASE_URL
}

consumption_daily = pd.read_csv("hourly_predictions.csv")

class MLTrader(Strategy):
    def initialize(self, data=consumption_daily, symbol:str="XLE", cash_at_risk:float=.5):
        self.data = data
        self.symbol = symbol
        self.sleeptime = "24H"
        self.last_trade = None
        self.current_day = 0
        self.cash_at_risk = cash_at_risk

    def position_sizing(self):
        cash = self.get_cash()
        last_price =  self.get_last_price(self.symbol)
        quantity = round(cash * self.cash_at_risk / last_price, 0)
        return cash, last_price, quantity

    def on_trading_iteration(self):
        if self.current_day >= len(self.data):
            self.should_continue = False
            return
        row = self.data.iloc[self.current_day]
        up_or_down = row["up_or_down"]
        cash, last_price, quantity = self.position_sizing()


        if up_or_down == 1:
            order = self.create_order(self.symbol, quantity, "buy", type="market")
            self.submit_order(order)
        elif up_or_down == 0:
            order = self.create_order(self.symbol, quantity, "sell", type="market")
            self.submit_order(order)


        self.current_day += 1

broker = Alpaca(ALPACA_CREDS)


strategy = MLTrader(name='mlstrat', broker=broker, parameters={
    "data": consumption_daily, "symbol": "XLE", "cash_at_risk": .5
})

start_date = datetime(2015, 1, 1)
end_date = datetime(2018, 7, 24)
strategy.backtest(YahooDataBacktesting, 
                  start_date, end_date, parameters={})

