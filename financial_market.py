import mesa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import random

#defining the Trader Agent
class Trader(Agent):
    def __init__(self, unique_id, model, strategy):
        super().__init__(unique_id, model)
        self.strategy = strategy  # Type of trader (momentum, value, noise, market_maker)
        self.cash = 10000  # Start with $10,000
        self.shares = 100  # Start with 100 shares
        self.portfolio_value = []  # Track portfolio value over time
        self.trade_history = []    # Track all trades

    def step(self):
        """Defines trader's action at each time step."""
        current_price = self.model.price  # Get the latest stock price
        shares_to_trade = 0

        # Decision based on strategy
        if self.strategy == "momentum":
            shares_to_trade = self.momentum_strategy()
        elif self.strategy == "value":
            shares_to_trade = self.value_strategy()
        elif self.strategy == "noise":
            shares_to_trade = self.noise_strategy()
        elif self.strategy == "market_maker":
            shares_to_trade = self.market_maker_strategy()

        self.place_order(shares_to_trade)  # Execute the trade
        
        # Update portfolio value
        self.portfolio_value.append(self.cash + self.shares * self.model.price)

    def momentum_strategy(self):
        """Follows the trend: Buy when price rises, sell when it falls."""
        price_history = self.model.price_history
        if len(price_history) < 2:
            return 0  # Not enough data
        if price_history[-1] > price_history[-2]:  # Price is increasing
            return random.randint(1, 5)  # Buy small amount
        else:
            return -random.randint(1, 5)  # Sell small amount

    def value_strategy(self):
        """Compares market price with the fundamental value and buys/sells accordingly."""
        diff = self.model.fundamental_value - self.model.price
        if diff > 5:  # Undervalued stock
            return random.randint(1, 10)  # Buy
        elif diff < -5:  # Overvalued stock
            return -random.randint(1, 10)  # Sell
        return 0

    def noise_strategy(self):
        """Trades randomly with no strategy."""
        return random.choice([-10, -5, 0, 5, 10])

    def market_maker_strategy(self):
        """Provides liquidity by buying when price is low and selling when high."""
        return random.choice([-3, -2, 0, 2, 3])

    def place_order(self, shares_to_trade):
        """Executes buy/sell order."""
        price = self.model.price
        cost = shares_to_trade * price
        
        # Execute trade if valid
        if shares_to_trade > 0 and self.cash >= cost:  # Buy order
            self.shares += shares_to_trade
            self.cash -= cost
            self.trade_history.append({"step": self.model.schedule.steps, 
                                      "action": "buy", 
                                      "shares": shares_to_trade, 
                                      "price": price})
            self.model.order_book.append(shares_to_trade)  # Add to order book
            
        elif shares_to_trade < 0 and self.shares >= abs(shares_to_trade):  # Sell order
            self.shares += shares_to_trade  # Subtract shares[hadasaab@archlinux MESA-GSOC]$ cd ~/ABM-Market-Simulation
cp -r financial-market-abm financial-market-abm-backup
bash: cd: /home/hadasaab/ABM-Market-Simulation: No such file or directory
cp: cannot stat 'financial-market-abm': No such file or directory
[hadasaab@archlinux MESA-GSOC]$ AAA


            self.cash -= cost  # Add proceeds (cost is negative)
            self.trade_history.append({"step": self.model.schedule.steps, 
                                      "action": "sell", 
                                      "shares": abs(shares_to_trade), 
                                      "price": price})
            self.model.order_book.append(shares_to_trade)  # Add to order book
