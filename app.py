import streamlit as st
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from pennylane import numpy as qnp
import pennylane as qml

def get_stock_data_batch(stocks, start_date, end_date, delay=1):
    """Fetch historical stock data in batches with delay."""
    stock_data = []
    for stock in stocks:
        try:
            df = yf.download(f"{stock}.NS", start=start_date, end=end_date)
            if not df.empty:
                df = df[['Close']].rename(columns={'Close': stock})
                stock_data.append(df)
                print(f"Data fetched for {stock}")
            else:
                print(f"No data available for {stock}")
        except Exception as e:
            print(f"Error fetching data for {stock}: {e}")
        
        # Delay to avoid hitting API rate limit
        time.sleep(delay)

    if not stock_data:
        raise ValueError("No valid stock data was fetched. Please check the stock symbols or date range.")

    final_data = pd.concat(stock_data, axis=1)
    return final_data

def create_cost_function_with_target_return(daily_returns, target_return, return_weight=10.0, negative_penalty=5.0):
    """Define the cost function for portfolio optimization."""
    covariance_matrix = np.cov(daily_returns.T)
    mean_returns = np.mean(daily_returns, axis=0)

    def cost_function(weights):
        weights = np.array(weights, dtype=np.float64)  # Ensure weights are in supported format
        weights = weights / np.sum(weights)  # Normalize weights

        # Portfolio return and risk
        portfolio_return = np.dot(mean_returns, weights)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))

        # Penalize negative returns
        negative_return_penalty = np.sum(np.maximum(-mean_returns, 0) * weights) * negative_penalty

        # Weighted cost function
        return_penalty = (portfolio_return - target_return) ** 2 * return_weight
        total_cost = portfolio_risk - portfolio_return + return_penalty + negative_return_penalty

        return total_cost

    return cost_function


def optimize_portfolio_with_penalty(daily_returns, target_return, num_stocks, return_weight=10.0, negative_penalty=5.0):
    """Optimize portfolio weights with penalties for negative returns."""
    dev = qml.device("default.qubit", wires=num_stocks)

    # Quantum node
    @qml.qnode(dev)
    def circuit(params):
        qml.templates.StronglyEntanglingLayers(params, wires=range(num_stocks))
        return [qml.expval(qml.PauliZ(i)) for i in range(num_stocks)]

    # Generate the cost function
    cost_function = create_cost_function_with_target_return(
        daily_returns, target_return, return_weight, negative_penalty
    )

    # Initial parameters
    num_layers = 2
    params = np.random.random((num_layers, num_stocks, 3))  # Initialize params as numpy array

    # Optimization loop
    optimizer = qml.GradientDescentOptimizer(stepsize=0.1)

    def wrapped_cost(v):
        weights = params_to_weights(circuit(v))
        return cost_function(weights)

    for i in range(200):
        params = optimizer.step(wrapped_cost, params)

    # Convert the optimized parameters into portfolio weights
    weights = params_to_weights(circuit(params))
    weights = weights / np.sum(weights)  # Normalize

    return weights


def params_to_weights(circuit_output):
    """Convert quantum circuit output into portfolio weights."""
    return np.abs(circuit_output)


def calculate_portfolio_performance(weights, daily_returns):
    """Calculate the portfolio return and risk based on the optimized weights."""
    portfolio_return = np.dot(np.mean(daily_returns, axis=0), weights)
    covariance_matrix = np.cov(daily_returns.T)
    portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))

    return portfolio_return, portfolio_risk


def plot_portfolio(weights, stock_names):
    """Plot the optimized portfolio as a pie chart."""
    plt.figure(figsize=(8, 6))
    plt.pie(weights, labels=stock_names, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
    plt.title("Optimized Portfolio Allocation")
    plt.show()


def calculate_user_portfolio_return(daily_returns, weights, total_investment):
    """Calculate the expected portfolio return based on user input investment."""
    mean_daily_returns = daily_returns.mean()
    portfolio_return = np.dot(weights, mean_daily_returns)
    total_portfolio_return = total_investment * portfolio_return
    return portfolio_return, total_portfolio_return


# Main code
if __name__ == "__main__":
    # List of available stock symbols for the user to choose from
    available_stock_symbols = [
        'RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK', 'SBIN', 'ADANIGREEN', 
        'BAJAJFINSV', 'MARUTI', 'HINDUNILVR', 'ITC', 'ASIANPAINT', 'WIPRO', 'M&M',
        'ONGC', 'TATAMOTORS', 'BAJAJFINANCE', 'LUPIN', 'HCLTECH', 'BHARTIARTL'
    ]
    
    # Print the available symbols for the user to choose
    print("Available stock symbols:")
    for idx, symbol in enumerate(available_stock_symbols, 1):
        print(f"{idx}. {symbol}")

    # User selects stocks by entering numbers corresponding to symbols
    selected_indexes = input("Enter the numbers corresponding to the stocks you want to include in your portfolio (comma-separated): ").split(",")
    selected_indexes = [int(index.strip()) - 1 for index in selected_indexes]  # Adjust for 0-indexing

    # Get the selected stock symbols
    selected_stocks = [available_stock_symbols[i] for i in selected_indexes]

    # User input for total investment
    total_investment = float(input("Enter your investment amount (in INR): "))
    start_date = "2022-11-01"
    end_date = "2023-11-01"
    delay_between_fetches = 2  # seconds to delay API calls

    print("Fetching stock data for the selected stocks...")
    stock_prices = get_stock_data_batch(selected_stocks, start_date, end_date, delay=delay_between_fetches)

    # Ensure no missing data (NaN values) are in the fetched data
    stock_prices = stock_prices.dropna(axis=1)  # Drop columns with NaN values
    
    # Calculate daily returns
    daily_returns = stock_prices.pct_change().dropna()
    mean_returns = daily_returns.mean()
    print("Mean Daily Returns:")
    print(mean_returns)

    # Optimize portfolio
    print("Optimizing portfolio...")
    target_return = 0.84 / 252  # Target daily return (84% annualized)
    optimized_weights = optimize_portfolio_with_penalty(
        daily_returns, target_return, len(selected_stocks), return_weight=10.0, negative_penalty=5.0
    )

    print("\nOptimized Portfolio Weights:")
    for stock, weight in zip(selected_stocks, optimized_weights):
        print(f"{stock}: {weight * 100:.2f}%")

    # Portfolio performance
    portfolio_return, portfolio_risk = calculate_portfolio_performance(optimized_weights, daily_returns)
    print(f"\nExpected Portfolio Return: {portfolio_return * 100:.2f}%")
    print(f"Expected Portfolio Risk (Volatility): {portfolio_risk * 100:.2f}%")

    # Plot portfolio
    plot_portfolio(optimized_weights, selected_stocks)

    # Calculate portfolio return with user investment
    _, total_portfolio_return = calculate_user_portfolio_return(daily_returns, optimized_weights, total_investment)
    print(f"\nTotal portfolio return on ₹{total_investment} investment: ₹{total_portfolio_return:.2f}")
