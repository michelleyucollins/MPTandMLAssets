import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Helper Functions for Portfolio Simulation and Frontier Computation


def simulate_portfolios(mean_returns, cov_matrix, num_portfolios=5000):
    """
    Randomly simulate a number of portfolios and return arrays of portfolio volatility,
    expected return, and Sharpe ratio (assuming risk-free rate = 0).
    """
    n = len(mean_returns)
    results = np.zeros((3, num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(n)
        weights /= np.sum(weights)
        weights_record.append(weights)
        port_return = np.sum(mean_returns * weights)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        results[0, i] = port_vol
        results[1, i] = port_return
        results[2, i] = (
            port_return / port_vol
        )  # Sharpe ratio (risk-free rate assumed 0)
    return results, weights_record


def compute_efficient_frontier(results):
    """
    From simulation results (volatilities and returns), compute the efficient frontier
    as the envelope of portfolios with highest returns at each level of risk.
    """
    vol = results[0]
    ret = results[1]
    sorted_idx = np.argsort(vol)
    vol_sorted = vol[sorted_idx]
    ret_sorted = ret[sorted_idx]

    frontier_vol = []
    frontier_ret = []
    current_max = -np.inf
    for v, r in zip(vol_sorted, ret_sorted):
        if r > current_max:
            current_max = r
            frontier_vol.append(v)
            frontier_ret.append(r)
    return np.array(frontier_vol), np.array(frontier_ret)


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
# The Magnificent 7
set3 = ["MSFT", "AMZN", "META", "AAPL", "GOOG", "NVDA", "TSLA"]


# Compute Efficient Frontier for a given asset set
def get_frontier_for_set(asset_set, assets_dict, num_portfolios=5000):
    """
    For the given list of asset tickers, compute the mean daily return and covariance matrix
    from their common return series, simulate random portfolios, and compute the efficient frontier.
    Returns: (frontier_volatilities, frontier_returns, all_results)
    """
    returns_df = get_returns(asset_set, assets_dict)
    mean_returns = returns_df.mean().values
    cov_matrix = returns_df.cov().values

    results, _ = simulate_portfolios(mean_returns, cov_matrix, num_portfolios)
    frontier_vol, frontier_ret = compute_efficient_frontier(results)
    return frontier_vol, frontier_ret, results


# Plotting: Create side-by-side plots for the three efficient frontiers

# Make sure the output directory exists
results_dir = "./results"
os.makedirs(results_dir, exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Frontier for GOVT and SPY
front_vol1, front_ret1, results1 = get_frontier_for_set(set1, assets)
axes[0].scatter(
    results1[0, :], results1[1, :], c="lightgray", s=10, label="Simulated Portfolios"
)
axes[0].plot(front_vol1, front_ret1, color="blue", lw=2, label="Efficient Frontier")
axes[0].set_title("Efficient Frontier: GOVT & SPY")
axes[0].set_xlabel("Volatility")
axes[0].set_ylabel("Expected Return")
axes[0].legend()

# Frontier for all assets
front_vol2, front_ret2, results2 = get_frontier_for_set(set2, assets)
axes[1].scatter(
    results2[0, :], results2[1, :], c="lightgray", s=10, label="Simulated Portfolios"
)
axes[1].plot(front_vol2, front_ret2, color="green", lw=2, label="Efficient Frontier")
axes[1].set_title("Efficient Frontier: All Assets")
axes[1].set_xlabel("Volatility")
axes[1].set_ylabel("Expected Return")
axes[1].legend()

# Frontier for Magnificent 7
front_vol3, front_ret3, results3 = get_frontier_for_set(set3, assets)
axes[2].scatter(
    results3[0, :], results3[1, :], c="lightgray", s=10, label="Simulated Portfolios"
)
axes[2].plot(front_vol3, front_ret3, color="red", lw=2, label="Efficient Frontier")
axes[2].set_title("Efficient Frontier: Magnificent Seven")
axes[2].set_xlabel("Volatility")
axes[2].set_ylabel("Expected Return")
axes[2].legend()

plt.tight_layout()

# Save the figure
save_path = os.path.join(results_dir, "efficient_frontiers.png")
plt.savefig(save_path)
plt.show()

print(f"Efficient frontier plots saved to {save_path}")
