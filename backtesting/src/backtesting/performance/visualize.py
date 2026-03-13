import argparse
import os
import pickle
import sys
from typing import Any

import pandas as pd
import plotly.express as px

from backtesting.performance.metrics import print_parameter_ranking_stats

DEFAULT_FIELDS = (
    "amount",
    "total_allocation",
    "total_distributions",
    "weights",
    "allocation_schedule",
    "initial_allocation",
    "yearly_allocation",
)


def _resolve_results_path(path: str) -> str:
    """Validates and normalizes the required results file path."""

    def _is_hashed_results_file(candidate_path: str) -> bool:
        """Returns whether a path matches results_<hash>.pkl naming."""
        filename = os.path.basename(candidate_path)
        return filename.startswith("results_") and filename.endswith(".pkl")

    normalized_path = os.path.abspath(path)
    if not _is_hashed_results_file(normalized_path):
        print(f"Expected a hashed results file like results_<hash>.pkl, got: {path}")
        sys.exit(-1)

    return normalized_path


def _load_results_file(results_file: str) -> dict:
    """Loads and validates the simulation results pickle file."""
    if not os.path.isfile(results_file):
        print(f"Results file not found: {results_file}")
        sys.exit(-1)

    try:
        with open(results_file, "rb") as file_handle:
            raw_results = pickle.load(file_handle)
    except (pickle.UnpicklingError, EOFError, ValueError, AttributeError):
        print(f"Results file is not a valid pickle payload: {results_file}")
        sys.exit(-1)
    except OSError as error:
        print(f"Could not read results file ({results_file}): {error}")
        sys.exit(-1)

    if not isinstance(raw_results, dict):
        print(f"Unexpected results payload type: {type(raw_results)}")
        sys.exit(-1)

    return raw_results


def _to_plottable(value: Any) -> Any:
    """Converts schedule-like / complex values into display-friendly forms."""
    if value is None:
        return None

    if hasattr(value, "value"):
        return value.value

    # Normalize numpy arrays (e.g. weights) into hashable values for grouping.
    if hasattr(value, "tolist"):
        list_value = value.tolist()
        return tuple(list_value) if isinstance(list_value, list) else list_value

    if isinstance(value, (tuple, list)):
        return str(tuple(value))

    return value


def _flatten_results(raw_results: dict) -> pd.DataFrame:
    """Flattens grouped top-N simulation output into row-wise records."""
    rows = []
    for test_date, top_n in raw_results.items():
        for rank, result in enumerate(top_n, start=1):
            if not isinstance(result, (tuple, list)):
                continue

            row = {
                "date": pd.to_datetime(test_date),
                "rank": rank,
            }
            for index, value in enumerate(result):
                key = DEFAULT_FIELDS[index] if index < len(DEFAULT_FIELDS) else f"parameter_{index}"
                row[key] = _to_plottable(value)
            rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["date", "rank"]).reset_index(drop=True)
    return df


def _parse_parameters(parameters: list[str] | None, fallback_parameter: str) -> list[str]:
    """Parses CLI parameters and supports comma-delimited values."""
    if not parameters:
        return [fallback_parameter]

    parsed = []
    for value in parameters:
        if not value:
            continue
        parsed.extend([token.strip() for token in value.split(",") if token.strip()])
    return parsed or [fallback_parameter]


def plot_ranked_results_against_parameter(results: pd.DataFrame, parameter: str) -> None:
    """Visualizes final portfolio value against a simulation parameter."""
    if parameter not in results.columns:
        available = ", ".join(results.columns)
        print(f'Parameter "{parameter}" not found. Available columns: {available}')
        return
    if "amount" not in results.columns:
        print('Missing "amount" column in simulation results; cannot plot rankings.')
        return

    plot_df = results[[parameter, "amount", "rank", "date"]].dropna(subset=[parameter, "amount"])
    if plot_df.empty:
        print(f'No data available for parameter "{parameter}".')
        return

    fig = px.box(
        plot_df,
        x=parameter,
        y="amount",
        color=plot_df["rank"].astype(str),
        points="all",
        hover_data=["date", "rank"],
        title=f"Final Portfolio Value by {parameter}",
        labels={
            "amount": "Final Portfolio Value",
            parameter: parameter,
            "color": "Rank",
        },
    )
    fig.update_layout(legend_title_text="Result Rank")
    fig.show()


def main() -> None:
    """Parses CLI arguments, loads results, and renders analysis output."""
    parser = argparse.ArgumentParser(description="Analyze simulation results from pickle output.")
    parser.add_argument("results_file", help="Path to results_<hash>.pkl file")
    parser.add_argument(
        "-p",
        "--parameter",
        default="allocation_schedule",
        help="Parameter to analyze/plot (default: allocation_schedule)",
    )
    parser.add_argument(
        "-P",
        "--parameters",
        nargs="+",
        help="One or more parameters (supports comma-separated values)",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Run statistical ranking analysis only (no charts)",
    )
    parser.add_argument(
        "--list-parameters",
        action="store_true",
        help="List parsed columns and exit",
    )
    args = parser.parse_args()

    results_file = _resolve_results_path(args.results_file)
    raw_results = _load_results_file(results_file)

    results = _flatten_results(raw_results)
    if results.empty:
        print(f"No result rows found in {results_file}")
        sys.exit(-1)

    print(f"Loaded {len(results):,} rows from {results_file}")

    if args.list_parameters:
        print(", ".join(results.columns))
        return

    parameters = _parse_parameters(args.parameters, args.parameter)
    for parameter in parameters:
        print_parameter_ranking_stats(results, parameter)
        if not args.stats_only:
            plot_ranked_results_against_parameter(results, parameter)


if __name__ == "__main__":
    main()
