import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Helper Functions for Portfolio Simulation and Frontier Computation

def efficient_frontier_quad_prog(mean_returns, cov_matrix, num_points=50):
    """
    Compute the efficient frontier using quadratic programming via scipy.optimize.minimize.
    For a given set of expected returns and covariance matrix, solve:
    
       minimize   0.5 * x.T * cov_matrix * x
       subject to sum(x) == 1,
                  x.T * mean_returns == target_return,
                  x_i >= 0  for all i.
    
    The function varies the target_return over a specified range.
    
    Returns:
      frontier_vol: Numpy array of portfolio volatilities along the frontier.
      frontier_ret: Numpy array of portfolio returns along the frontier.
      weights_list: List of optimal weight vectors for each target.
    """
    n = len(mean_returns)
    
    # Objective function: portfolio variance (note: we use 0.5 * variance)
    def objective(x):
        return 0.5 * np.dot(x, np.dot(cov_matrix, x))
    
    # Its gradient
    def objective_grad(x):
        return np.dot(cov_matrix, x)
    
    # Common equality constraint: sum(x)=1.
    constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1.0}]
    # Bounds: no short sales allowed.
    bounds = [(0, 1) for _ in range(n)]
    
    # Define a range of target returns.
    r_min = np.min(mean_returns)
    r_max = np.max(mean_returns)
    target_returns = np.linspace(r_min, r_max, num_points)
    
    frontier_vol = []
    frontier_ret = []
    weights_list = []
    
    for r_target in target_returns:
        # Add the target return equality constraint.
        cons = constraints + [
            {"type": "eq", "fun": lambda x, r_target=r_target: np.dot(x, mean_returns) - r_target}
        ]
        # Initial guess: equal weight.
        x0 = np.ones(n) / n
        res = minimize(objective, x0, method="SLSQP", jac=objective_grad,
                       bounds=bounds, constraints=cons)
        if not res.success:
            # Optionally, one could try another initial guess or increase iterations.
            print(f"Optimization failed for target return {r_target:.4f}: {res.message}")
            continue
        x_opt = res.x
        port_return = np.dot(x_opt, mean_returns)
        port_vol = np.sqrt(np.dot(x_opt, np.dot(cov_matrix, x_opt)))
        frontier_vol.append(port_vol)
        frontier_ret.append(port_return)
        weights_list.append(x_opt)
    return np.array(frontier_vol), np.array(frontier_ret), weights_list

# Data Loading and Return Computation

SPY = pd.read_csv("./data/SPY.csv", parse_dates=["Date"], index_col="Date")
GOVT = pd.read_csv("./data/GOVT.csv", parse_dates=["Date"], index_col="Date")
MSFT = pd.read_csv("./data/MSFT.csv", parse_dates=["Date"], index_col="Date")
AMZN = pd.read_csv("./data/AMZN.csv", parse_dates=["Date"], index_col="Date")
META = pd.read_csv("./data/META.csv", parse_dates=["Date"], index_col="Date")
AAPL = pd.read_csv("./data/AAPL.csv", parse_dates=["Date"], index_col="Date")
GOOG = pd.read_csv("./data/GOOG.csv", parse_dates=["Date"], index_col="Date")
NVDA = pd.read_csv("./data/NVDA.csv", parse_dates=["Date"], index_col="Date")
TSLA = pd.read_csv("./data/TSLA.csv", parse_dates=["Date"], index_col="Date")

# Dictionary for easy access
assets = {
    "GOVT": GOVT,
    "SPY": SPY,
    "MSFT": MSFT,
    "AMZN": AMZN,
    "META": META,
    "AAPL": AAPL,
    "GOOG": GOOG,
    "NVDA": NVDA,
    "TSLA": TSLA,
}


def get_returns(asset_list, assets_dict):
    """
    Given a list of asset tickers and a dictionary of asset DataFrames (with Date index and a 'Close' column),
    this function returns a DataFrame of daily percentage returns for the common dates.
    """
    df_list = []
    for ticker in asset_list:
        df = assets_dict[ticker][["Close"]].rename(columns={"Close": ticker})
        df_list.append(df)
    df_all = pd.concat(df_list, axis=1)
    df_all = df_all.dropna()  # only keep dates where every asset has data
    returns = df_all.pct_change().dropna()
    return returns


# Define the three asset sets
# Only GOVT and SPY
set1 = ["GOVT", "SPY"]
# All assets named
set2 = ["GOVT", "SPY", "MSFT", "AMZN", "META", "AAPL", "GOOG", "NVDA", "TSLA"]
# The Magnificent 7 + GOVT
set3 = ["GOVT", "MSFT", "AMZN", "META", "AAPL", "GOOG", "NVDA", "TSLA"]


# Compute Efficient Frontier for a given asset set
def get_frontier_for_set(asset_set, assets_dict, num_points=50):
    """
    For a given list of asset tickers, compute the common return series,
    determine the mean returns and covariance matrix, and then compute
    the efficient frontier using quadratic programming.
    
    Returns:
      frontier_vol: portfolio volatilities along the frontier.
      frontier_ret: portfolio returns along the frontier.
      weights_list: list of weight vectors (one for each frontier point).
    """
    returns_df = get_returns(asset_set, assets_dict)
    mean_returns = returns_df.mean().values
    cov_matrix = returns_df.cov().values
    frontier_vol, frontier_ret, weights_list = efficient_frontier_quad_prog(mean_returns, cov_matrix, num_points)
    return frontier_vol, frontier_ret, weights_list


# Plotting: Create side-by-side plots for the three efficient frontiers

# Make sure the output directory exists
results_dir = "./results"
os.makedirs(results_dir, exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Frontier for GOVT and SPY
front_vol1, front_ret1, _ = get_frontier_for_set(set1, assets)
axes[0].plot(front_vol1, front_ret1, color="blue", lw=2, label="Efficient Frontier")
axes[0].set_title("Efficient Frontier: GOVT & SPY")
axes[0].set_xlabel("Volatility")
axes[0].set_ylabel("Expected Return")
axes[0].legend()

# Frontier for all assets
front_vol2, front_ret2, _ = get_frontier_for_set(set2, assets)
axes[1].plot(front_vol2, front_ret2, color="green", lw=2, label="Efficient Frontier")
axes[1].set_title("Efficient Frontier: All Assets")
axes[1].set_xlabel("Volatility")
axes[1].set_ylabel("Expected Return")
axes[1].legend()

# Frontier for Magnificent 7
front_vol3, front_ret3, _ = get_frontier_for_set(set3, assets)
axes[2].plot(front_vol3, front_ret3, color="red", lw=2, label="Efficient Frontier")
axes[2].set_title("Efficient Frontier: Magnificent Seven + GOVT")
axes[2].set_xlabel("Volatility")
axes[2].set_ylabel("Expected Return")
axes[2].legend()

plt.tight_layout()

# Save the figure.
save_path = os.path.join(results_dir, "efficient_frontiers.png")
plt.savefig(save_path)
plt.show()

print(f"Efficient frontier plots saved to {save_path}")
