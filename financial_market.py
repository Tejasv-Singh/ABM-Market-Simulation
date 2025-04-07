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

class StockMarket(Model):
    """Model for a simple stock market with different types of traders"""
    
    def __init__(self, num_momentum=25, num_value=25, num_noise=25, num_market_makers=5, 
                 initial_price=100, volatility=0.01, mean_reversion=0.005):
        self.num_agents = num_momentum + num_value + num_noise + num_market_makers
        self.schedule = RandomActivation(self)
        self.price = initial_price
        self.price_history = [initial_price]
        self.fundamental_value = initial_price  # Initial fundamental value
        self.volatility = volatility
        self.mean_reversion = mean_reversion
        self.order_book = []  # To track all orders
        self.volume_history = []  # Trading volume by step
        
        # Create traders of different types
        agent_id = 0
        # Momentum traders
        for i in range(num_momentum):
            a = Trader(agent_id, self, "momentum")
            self.schedule.add(a)
            agent_id += 1
            
        # Value traders
        for i in range(num_value):
            a = Trader(agent_id, self, "value")
            self.schedule.add(a)
            agent_id += 1
            
        # Noise traders
        for i in range(num_noise):
            a = Trader(agent_id, self, "noise")
            self.schedule.add(a)
            agent_id += 1
            
        # Market makers
        for i in range(num_market_makers):
            a = Trader(agent_id, self, "market_maker")
            self.schedule.add(a)
            agent_id += 1
        
        # Add data collector
        self.datacollector = DataCollector(
            model_reporters={"Price": lambda m: m.price,
                            "FundamentalValue": lambda m: m.fundamental_value,
                            "Volume": lambda m: sum(abs(x) for x in m.order_book)},
            agent_reporters={"Cash": lambda a: a.cash,
                            "Shares": lambda a: a.shares,
                            "PortfolioValue": lambda a: a.cash + a.shares * a.model.price}
        )
    
    def update_price(self):
        """Update stock price based on order imbalance and random noise"""
        if not self.order_book:  # If no orders, add small random walk
            price_change = np.random.normal(0, self.volatility * self.price)
        else:
            # Net order flow (positive: more buys, negative: more sells)
            net_order_flow = sum(self.order_book)
            
            # Price impact from order imbalance
            impact = 0.1 * net_order_flow / len(self.order_book)
            
            # Random noise component
            noise = np.random.normal(0, self.volatility * self.price)
            
            # Mean reversion to fundamental value
            reversion = self.mean_reversion * (self.fundamental_value - self.price)
            
            # Combined price change
            price_change = impact + noise + reversion
            
        # Update price with constraints to prevent negative prices
        self.price = max(0.1, self.price + price_change)
        self.price_history.append(self.price)
        
        # Record volume
        volume = sum(abs(x) for x in self.order_book)
        self.volume_history.append(volume)
        
        # Clear order book for next step
        self.order_book = []
    
    def update_fundamental_value(self):
        """Occasionally update the fundamental value with random shocks"""
        if random.random() < 0.05:  # 5% chance of news event
            shock = np.random.normal(0, 5)  # Random news shock
            self.fundamental_value = max(10, self.fundamental_value + shock)
    
    def step(self):
        """Advance the model by one step"""
        # Collect data before the step
        self.datacollector.collect(self)
        
        # Execute all agent actions
        self.schedule.step()
        
        # Update fundamental value
        self.update_fundamental_value()
        
        # Update stock price based on trades
        self.update_price()


# Functions for analysis and visualization

def run_market_simulation(steps=200, num_momentum=25, num_value=25, 
                         num_noise=25, num_market_makers=5):
    """Run a complete market simulation and return the model"""
    model = StockMarket(num_momentum=num_momentum, 
                        num_value=num_value, 
                        num_noise=num_noise, 
                        num_market_makers=num_market_makers)
    
    for i in range(steps):
        model.step()
    
    return model

def analyze_market_data(model):
    """Analyze results from a market simulation"""
    # Get model-level data
    model_data = model.datacollector.get_model_vars_dataframe()
    
    # Get agent-level data
    agent_data = model.datacollector.get_agent_vars_dataframe()
    
    # Calculate returns
    model_data['Returns'] = model_data['Price'].pct_change()
    
    # Calculate volatility
    volatility = model_data['Returns'].std() * np.sqrt(252)  # Annualized
    
    # Analyze by trader type
    trader_types = {'momentum': [], 'value': [], 'noise': [], 'market_maker': []}
    for agent in model.schedule.agents:
        trader_types[agent.strategy].append(agent.unique_id)
    
    # Calculate performance by trader type
    performance = {}
    for strategy, agent_ids in trader_types.items():
        if agent_ids:
            # Get portfolio values for this strategy
            strategy_data = agent_data.xs(agent_ids[0], level="AgentID")['PortfolioValue']
            
            # Calculate average final portfolio value for this strategy
            performance[strategy] = sum(agent.portfolio_value[-1] 
                                      for agent in model.schedule.agents 
                                      if agent.strategy == strategy) / len(agent_ids)
    
    results = {
        'final_price': model_data['Price'].iloc[-1],
        'volatility': volatility,
        'trading_volume': sum(model.volume_history),
        'performance_by_type': performance
    }
    
    return results, model_data, agent_data

def plot_market_results(model_data):
    """Plot key market metrics from simulation"""
    plt.figure(figsize=(15, 10))
    
    # Price and fundamental value
    plt.subplot(3, 1, 1)
    plt.plot(model_data['Price'], label='Market Price')
    plt.plot(model_data['FundamentalValue'], label='Fundamental Value', linestyle='--')
    plt.title('Stock Price vs Fundamental Value')
    plt.legend()
    
    # Trading volume
    plt.subplot(3, 1, 2)
    plt.bar(model_data.index, model_data['Volume'], alpha=0.7)
    plt.title('Trading Volume')
    
    # Returns distribution
    plt.subplot(3, 1, 3)
    plt.hist(model_data['Returns'].dropna(), bins=30, alpha=0.7)
    plt.title('Distribution of Returns')
    
    plt.tight_layout()
    plt.show()

def compare_trader_performance(model):
    """Compare performance of different trader types"""
    plt.figure(figsize=(10, 6))
    
    # Get a trader of each type
    trader_samples = {}
    for agent in model.schedule.agents:
        if agent.strategy not in trader_samples:
            trader_samples[agent.strategy] = agent
    
    # Plot portfolio values over time
    for strategy, agent in trader_samples.items():
        plt.plot(agent.portfolio_value, label=strategy)
    
    plt.title('Portfolio Value by Trader Type')
    plt.xlabel('Steps')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Example of running a simulation
if __name__ == "__main__":
    # Run simulation
    market_model = run_market_simulation(steps=200)
    
    # Analyze results
    results, model_data, agent_data = analyze_market_data(market_model)
    
    # Display summary
    print("Final price:", results['final_price'])
    print("Annualized volatility:", results['volatility'])
    print("Total trading volume:", results['trading_volume'])
    print("\nPerformance by trader type:")
    for strategy, performance in results['performance_by_type'].items():
        print(f"{strategy}: ${performance:.2f}")
    
    # Plot results
    plot_market_results(model_data)
    compare_trader_performance(market_model)