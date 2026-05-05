# Backtesting

A Python backtesting toolkit for running portfolio allocation/rebalancing tests against historical market data from Yahoo Finance.

It supports:
- Single strategy test runs
- Parallelized parameter simulations
- Strategy schedules and allocation schedules (static or dynamic)
- Post-run analysis/visualization for simulation result files

## Repository Layout

- `config.yaml` - Example configuration file for test runs
- `backtesting/pyproject.toml` - Python package metadata and dependencies
- `backtesting/__main__.py` - CLI entrypoint for running tests/simulations
- `backtesting/src/backtesting/core/` - Portfolio engine, scheduling, simulation, utilities
- `backtesting/src/backtesting/strategies/` - Strategy implementations
- `backtesting/src/backtesting/performance/visualize.py` - Simulation result analysis and plotting

## Requirements

- Python 3.10+ recommended
- Internet access for initial ticker data download (`yfinance`)

## Installation

From the repository root:

```bash
python -m pip install -e ./backtesting
```

## Quick Start

1. Edit `config.yaml` (an example `simulate` config is already included).
2. Run:

```bash
python -m backtesting config.yaml
```

## What Gets Created

When you run tests from a directory, artifacts are written to that working directory:

- `tickers/*.pkl` - Cached historical ticker data
- `simulation_parameters_<hash>.pkl` - Cached simulation parameter axes
- `checkpoint_<hash>.txt` - Simulation progress checkpoint
- `results_<hash>.pkl` - Top-ranked simulation outputs grouped by test date

## Running Test Types

The top-level config key is `test`, with one or both modes:

- `single`: deterministic single run(s) with a specific strategy + config
- `simulate`: parameter sweep using multiprocessing and top-N result retention

Example structure:

```yaml
test:
  single:
    - ...
  simulate:
    - ...
```

## Configuration Reference

Each test item under `single` or `simulate` accepts these blocks:

- `securities` (required)
  - List of `{ticker, weight}` entries
  - Weights must sum to `1.0`
  - Optional synthetic-return fields per security:
    - `daily_return_multiplier`: float leverage factor (for example `2.0` or `3.0`)
    - `underlying_ticker`: source ticker used to build synthetic prices (defaults to `ticker`)
    - `expense_ratio`: annual expense ratio as a decimal float (defaults to `0.0`)
- `strategy` (required)
  - `name`: strategy class name
  - Additional strategy-specific args
- `allocation` (required)
  - `initial`: initial capital (numeric, > 0)
  - `yearly`: annual contribution (integer, >= 0)
  - `mode`: `static` or `dynamic`
  - Optional schedule config for static mode
- `weights` (optional)
  - `mode`: `static` or `dynamic` (dynamic is simulation-only)
  - `increment`: optional step size for dynamic weights
- `dates` (optional)
  - `start`: ISO date string (`YYYY-MM-DD`)
  - `end`: ISO date string (`YYYY-MM-DD`)
- `performance` (optional)
  - `benchmark`: benchmark ticker/index used for beta, Jensen's Alpha, and Treynor
  - `risk_free_ticker`: ticker used to estimate risk-free rate (`^IRX` recommended)
  - Required for `single` tests
- `trace` (optional)
  - Enables detailed transaction output during execution
- `track_performance` (optional)
  - For `single` tests this is always enabled so performance metrics can be reported

### Schedule Formats

Supported schedule formats:

- `DAYS` (integer `value` expected)
- `WEEKDAY` (weekday string like `Monday`)
- `WEEKLY`
- `MONTHLY`
- `YEARLY`

Used in:
- `allocation.schedule` (static allocation mode)
- `strategy.schedule` (when strategy supports schedule-based behavior)

### Synthetic Leveraged Securities

If `daily_return_multiplier` is provided for a security, the engine generates synthetic
open/close prices from the selected underlying ticker before running the backtest.

Synthetic price construction:

- Daily expense drag:
  - `daily_expense_rate = expense_ratio / 252`
- Close-to-close move:
  - `synthetic_close = previous_synthetic_close * (1 + underlying_daily_return * multiplier - daily_expense_rate)`
- Overnight open move:
  - `synthetic_open = previous_synthetic_close * (1 + underlying_overnight_return * multiplier)`
- Daily financing drag for borrowed exposure (see below):
  - `effective_borrowing_size = max(daily_return_multiplier - 1.0, 0.0)`
  - `daily_financing_rate = annual_funding_rate / 365`
  - `financing_drag = synthetic_open * daily_financing_rate * effective_borrowing_size`

The synthetic close series is then used by the current portfolio execution logic.

### Synthetic Financing Costs

Leveraged synthetic securities (those with `daily_return_multiplier > 1.0`) imply
borrowed exposure equal to `daily_return_multiplier - 1` per dollar of NAV. The
engine charges a daily borrowing cost using a configurable overnight funding rate
sourced from FRED. By default it uses the daily Effective Federal Funds Rate
(`DFF`), which provides daily history back to 1954.

Annual percent rates from FRED are converted to a daily decimal via
`annual_rate_percent / 100 / 365`, then multiplied by the synthetic position's
borrowed size and current NAV to produce the dollar drag that is subtracted from
each synthetic close.

Configuration is optional and defaults are sensible. To override or disable:

```yaml
financing:
  enabled: true        # set to false to skip financing drag
  rate_source: fred    # only "fred" is supported today
  series_id: DFF       # any daily FRED series id (e.g. SOFR for post-2018 only)
```

Cached FRED data is stored under `tickers/fred/<SERIES_ID>.pkl`. Delete that file
to refresh the rate history on the next run.

## Strategy Names

Available strategies in `backtesting/src/backtesting/strategies/rebalance.py`:

- `ScheduledRebalance`
- `SimpleMovingAverageRebalance`
- `StdDevRebalance`
- `SMACrossRebalance`
- `VolatilityAdjustedSMARebalance`

Strategy-specific fields (such as `target`, `primary`, `alternate`, SMA lengths, etc.) go under the `strategy` block.

### Contribution weights vs rebalance weights

`securities[].weight` is the default portfolio mix and the target used on scheduled rebalances. Optional overrides control how **new** cash is invested (initial and periodic allocations) and how **distributions** are reinvested:

- `allocation.weights`: map each portfolio ticker to a weight (must sum to `1.0`). If omitted, contributions follow `securities[].weight`.
- `distribution.weights`: same shape; if omitted, distributions use `allocation.weights` when that is set, otherwise `securities[].weight`.

Example: put new capital and dividends 100% into `TQQQ` while rebalancing to 90% `TQQQ` / 10% `QQQI`:

```yaml
securities:
  - ticker: TQQQ
    weight: 0.9
  - ticker: QQQI
    weight: 0.1
allocation:
  initial: 10000
  yearly: 10000
  weights:
    TQQQ: 1.0
    QQQI: 0.0
distribution:
  weights:
    TQQQ: 1.0
    QQQI: 0.0
```

## Example `simulate` Configuration

```yaml
test:
  simulate:
    - securities:
        - ticker: TQQQ
          weight: 1.0
        - ticker: QQQ
          weight: 0.0
      strategy:
        name: ScheduledRebalance
        schedule:
          mode: static
          format: DAYS
          value: 7
      allocation:
        mode: dynamic
        increment: 4
        initial: 10000
        yearly: 6000
      weights:
        mode: static
      dates:
        start: "2020-01-01"
      performance:
        benchmark: ^NDX
        risk_free_ticker: ^IRX
      trace: false
```

For single tests, performance output includes CAGR, Sharpe, Sortino, Treynor, Jensen's Alpha, and beta.
Single-test configs must provide both `performance.benchmark` and
`performance.risk_free_ticker`. The risk-free ticker is interpreted like `^IRX`
(close values are yield percentages) and converted to an annualized decimal rate.

## Visualizing Simulation Results

Use the visualization utility with a generated results file:

```bash
python -m backtesting.performance.visualize results_<hash>.pkl
```

Useful options:

- `--list-parameters` - Print available result columns and exit
- `--stats-only` - Print ranking stats without opening charts
- `-p <parameter>` - Analyze one parameter
- `-P <p1> <p2>` - Analyze multiple parameters (also supports comma-separated values)

Example:

```bash
python -m backtesting.performance.visualize results_abc123def456.pkl --stats-only -P allocation_schedule weights
```

## Notes

- First run may take longer due to historical data download and cache creation.
- Simulations are resumable through checkpoint/result files with matching config hash.
- If a ticker download fails, execution exits with a clear error message.
