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
        self.distribution_history = {}
        self.unallocated_capital = 0.0

    def current_value(self, price_data: np.ndarray) -> float:
        """Returns the current total market value of all holdings."""
        return np.sum(self.current_shares * price_data)

    def record_distribution(self, date: object, amount: float) -> None:
        """Records portfolio-level distributions paid on a specific date."""
        self.distribution_history[date] = self.distribution_history.get(date, 0.0) + float(amount)

    def allocate(
        self,
        date_string: str,
        price_data: np.ndarray,
        amount: float,
        weights: tuple[float, ...],
        trace: bool = False,
    ) -> None:
        """Performs an allocation to all tickers on a specific date.

        If ``contribution_weights`` is set, that mix is used for this contribution only
        (rebalance target weights on the portfolio are unchanged).
        """

        # Check if weights were provided
        if weights is None:
            weights = self.weights

        # Account for new capital
        self.unallocated_capital += amount

        # Don't allocate anything less than $1
        if amount < 1.0:
            return

        # Get target ticker values
        ticker_values = (np.array(weights) * self.unallocated_capital).round(decimals=2)

        # Convert values to shares
        ticker_share_deltas = ticker_values / price_data
        ticker_share_deltas = ticker_share_deltas.round(decimals=4)

        # Add share deltas (subtract if negative)
        self.current_shares += ticker_share_deltas

        # Subtract the actual amount allocated from the unallocated capital
        self.unallocated_capital = max(0.0, (self.unallocated_capital - np.sum(ticker_share_deltas * price_data).round(decimals=2)))

        if trace:
            print(f"Allocated ${np.sum(ticker_values):,.2f} -- " + ", ".join(f"${ticker_values[i]:,.2f} to {self.tickers[i]}" for i in range(len(self.tickers))) + f" on {date_string}")

    def rebalance(
        self,
        date_string: str,
        price_data: np.ndarray,
        weights: tuple[float, ...] | None = None,
        trace: bool = False,
    ) -> None:
        """Performs a rebalance on all tickers in a portfolio."""

        # Use existing weights if not provided
        if weights is None:
            weights = self.weights

        # Get the current value of the portfolio
        current_portfolio_value = np.sum(self.current_shares * price_data).round(decimals=2)

        # Get current ticker values
        current_ticker_values = self.current_shares * price_data

        # Get target ticker values
        target_ticker_values = weights * current_portfolio_value

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

        if trace:
            print("Rebalanced " + ", ".join(f"{self.tickers[i]}: {'+' if filtered_ticker_value_deltas[i] > 0.0 else '-'}${abs(filtered_ticker_value_deltas[i]):,.2f}" for i in range(len(self.tickers))) + f" on {date_string}")
