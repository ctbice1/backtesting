import numpy as np

class Portfolio:
    """Stateful portfolio model used during backtest execution."""

    def __init__(self, tickers: tuple[str, ...], weights: tuple[float, ...], track: bool = False) -> None:
        """Initializes portfolio holdings, weights, and execution totals."""
        self.tickers: tuple[str] = tickers
        self.weights: tuple[float] = np.array(weights).round(decimals=4)
        self.ticker_idx: dict[str,int] = {tickers[i]: i for i in range(len(tickers))}
        self.current_shares: np.array = np.zeros(len(tickers))

        self.track = track
        self.total_new_capital = 0.0
        self.total_distribution = 0.0
        self.unallocated_capital = 0.0

    def current_value(self, price_data: np.ndarray) -> float:
        """Returns the current total market value of all holdings."""
        return np.sum(self.current_shares * price_data)

    def allocate(
        self, date_string: str, price_data: np.ndarray, amount: float, details: bool = False
    ) -> None:
        """Performs an allocation to all tickers on a specific date."""

        # Don't allocate anything less than $1
        if amount < 1.0:
            self.unallocated_capital += amount
            return

        # Get available capital
        available_capital = amount + self.unallocated_capital

        # Add the new amount to the current value of the portfolio
        current_portfolio_value = np.sum(self.current_shares * price_data).round(decimals=2)
        new_portfolio_value = current_portfolio_value + available_capital

        # Get ticker value deltas
        new_ticker_values = (self.weights * new_portfolio_value).round(decimals=2)
        current_ticker_values = (self.weights * current_portfolio_value).round(decimals=2)
        ticker_value_deltas = new_ticker_values - current_ticker_values

        # Filter any value deltas under $1
        filtered_ticker_value_deltas = np.where(ticker_value_deltas >= 1.0, ticker_value_deltas, 0.0)

        # Check if an allocation is needed
        required_capital = np.sum(filtered_ticker_value_deltas)
        if required_capital == 0.0:
            return

        # Remove any excess required capital
        if required_capital > available_capital:
            excess_amount = (required_capital - available_capital)
            weights = filtered_ticker_value_deltas / required_capital
            excess_amounts = (excess_amount * weights).round(decimals=2)
            filtered_ticker_value_deltas = filtered_ticker_value_deltas - excess_amounts

        # Convert values to shares
        ticker_share_deltas = filtered_ticker_value_deltas / price_data
        ticker_share_deltas = ticker_share_deltas.round(decimals=3)

        # Subtract the actual amount allocated from the allocation amount
        available_capital -= np.sum(filtered_ticker_value_deltas)

        # Add the remaining amount to unallocated capital
        self.unallocated_capital = available_capital

        # Add share deltas (subtract if negative)
        self.current_shares += ticker_share_deltas

        if details:
            print(f"Allocated ${np.sum(filtered_ticker_value_deltas):,.2f} -- " + ", ".join(f"${filtered_ticker_value_deltas[i]:,.2f} to {self.tickers[i]}" for i in range(len(self.tickers))) + f" on {date_string}")

    def rebalance(
        self,
        date_string: str,
        price_data: np.ndarray,
        target_weights: tuple[float, ...] | None = None,
        details: bool = False,
    ) -> None:
        """Performs a rebalance on all tickers in a portfolio."""

        # Use custom weights if provided
        if target_weights is not None:
            self.weights = target_weights

        # Get the current value of the portfolio
        current_portfolio_value = np.sum(self.current_shares * price_data).round(decimals=2)

        # Get current ticker values
        current_ticker_values = self.current_shares * price_data

        # Get target ticker values
        target_ticker_values = self.weights * current_portfolio_value

        # Remove any excess required capital
        target_portfolio_value = np.sum(target_ticker_values)
        if target_portfolio_value > current_portfolio_value:
            excess_amount = (target_portfolio_value - current_portfolio_value)
            weights = target_ticker_values / target_portfolio_value
            excess_amounts = (excess_amount * weights).round(decimals=2)
            target_ticker_values = target_ticker_values - excess_amounts

        # Get ticker value deltas >= $1 or <= -$1
        ticker_value_deltas = target_ticker_values - current_ticker_values
        filtered_ticker_value_deltas = np.where((ticker_value_deltas >= 1.0) | (ticker_value_deltas <= -1.0), ticker_value_deltas, 0.0)

        # Return if value deltas are insignificant to rebalance
        if np.all(filtered_ticker_value_deltas == 0.0):
            return

        # Get capital available for rebalancing
        available_capital = -np.sum(filtered_ticker_value_deltas[filtered_ticker_value_deltas <= -1.0])

        # If available capital exceeds required amount, adjust required amount down
        required_capital = np.sum(filtered_ticker_value_deltas[filtered_ticker_value_deltas >= 1.0])
        if required_capital > available_capital:
            excess_amount = (required_capital - available_capital)
            weights = filtered_ticker_value_deltas / required_capital
            excess_amounts = (excess_amount * weights).round(decimals=2)
            filtered_ticker_value_deltas = filtered_ticker_value_deltas - excess_amounts

        # Convert values to shares
        ticker_share_deltas = filtered_ticker_value_deltas / price_data
        ticker_share_deltas = ticker_share_deltas.round(decimals=3)

        # Add share deltas (subtract if negative)
        self.current_shares += ticker_share_deltas

        if details:
            print("Rebalanced " + ", ".join(f"{self.tickers[i]}: {'+' if filtered_ticker_value_deltas[i] > 0.0 else '-'}${abs(filtered_ticker_value_deltas[i]):,.2f}" for i in range(len(self.tickers))) + f" on {date_string}")
