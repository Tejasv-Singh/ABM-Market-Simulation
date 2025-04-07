# Agent-Based Stock Market Simulation

## Overview
This project implements an agent-based model (ABM) of a stock market where various types of traders interact to form market prices. The simulation demonstrates how market dynamics emerge from the interactions of different trading strategies and can be used to study market efficiency, price formation, bubbles, crashes, and trader performance.

## Features
- **Multiple trader types** with unique strategies:
  - Momentum traders: Follow price trends
  - Value traders: Trade based on deviation from fundamental value
  - Noise traders: Trade randomly
  - Market makers: Provide liquidity
- **Dynamic price formation** mechanism based on:
  - Order imbalance
  - Mean reversion to fundamental value
  - Random market noise
- **Comprehensive data collection**:
  - Price history
  - Fundamental value evolution
  - Trading volume
  - Agent-level portfolio values
- **Analysis tools** for evaluating:
  - Price dynamics
  - Market volatility
  - Returns distribution
  - Performance comparison between strategies

## Requirements
- Python 3.6+
- mesa
- numpy
- pandas
- matplotlib

## Installation
```bash
pip install mesa numpy pandas matplotlib
```

## Usage

### Basic Simulation
```python
from market_model import run_market_simulation, analyze_market_data, plot_market_results

# Run a simulation with default parameters
market_model = run_market_simulation(steps=200)

# Analyze the results
results, model_data, agent_data = analyze_market_data(market_model)

# Visualize the results
plot_market_results(model_data)
compare_trader_performance(market_model)
```

### Custom Parameters
```python
# Run with custom trader composition
market_model = run_market_simulation(
    steps=300,
    num_momentum=40,  # More momentum traders
    num_value=20,
    num_noise=10,
    num_market_makers=5
)
```

## Model Parameters

### StockMarket Parameters
- `num_momentum`: Number of momentum traders (default: 25)
- `num_value`: Number of value traders (default: 25)
- `num_noise`: Number of noise traders (default: 25)
- `num_market_makers`: Number of market makers (default: 5)
- `initial_price`: Starting price of the stock (default: 100)
- `volatility`: Market volatility parameter (default: 0.01)
- `mean_reversion`: Rate of mean reversion to fundamental value (default: 0.005)

### Trading Strategies

#### Momentum Strategy
Momentum traders follow trends by:
- Buying when prices are rising
- Selling when prices are falling

#### Value Strategy
Value traders attempt to profit from price deviations from fundamental value:
- Buying when price is below fundamental value
- Selling when price is above fundamental value

#### Noise Strategy
Noise traders make random trading decisions without any specific strategy.

#### Market Maker Strategy
Market makers provide liquidity by buying and selling in small amounts.

## Analysis Functions

### analyze_market_data()
Returns detailed metrics about the simulation:
- Final price
- Price volatility
- Trading volume
- Performance by trader type

### plot_market_results()
Generates visualizations of:
- Price vs. fundamental value over time
- Trading volume over time
- Distribution of returns

### compare_trader_performance()
Visualizes the performance of different trader types throughout the simulation.

## Extending the Model
The model can be extended in several ways:
- Add new trader strategies
- Implement more sophisticated order matching
- Include transaction costs or market friction
- Add multiple assets for portfolio allocation
- Implement regulatory mechanisms

## Example Results
Running the simulation will typically reveal:
1. How trading strategies perform in different market conditions
2. The emergence of market inefficiencies and their correction
3. The impact of trader composition on market stability
4. Patterns in price formation and volatility clustering

## License
This project is open source and available under the MIT License.
