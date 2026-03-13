from collections import defaultdict
from functools import partial
import heapq
import hashlib
import json
from multiprocessing import Pool, shared_memory
import os
import pickle
import sys
from typing import Iterator

import numpy as np

from backtesting.core.schedule import Schedule, ScheduleFormat
from backtesting.core.portfolio import Portfolio
from backtesting.core.util import (
    bound_to_available_dates,
    coerce_date,
    get_historical_data,
    top_n_grouped_incremental,
)

_WORKER_PRICE_DATA = None
_WORKER_DISTRIBUTION_DATA = None
_WORKER_PRICE_DATES = None
_WORKER_DISTRIBUTION_DATES = None
_WORKER_PRICE_SHM = None
_WORKER_DISTRIBUTION_SHM = None
_WORKER_PRICE_DATES_SHM = None
_WORKER_DISTRIBUTION_DATES_SHM = None

def _is_heap_result_item(item: object) -> bool:
    """Returns True when an item is in (score, tiebreaker, payload) heap format."""
    return (
        isinstance(item, (tuple, list))
        and len(item) == 3
        and isinstance(item[2], (tuple, list))
    )

def _normalize_grouped_results_for_incremental(raw_results: dict) -> defaultdict:
    """
    Normalizes persisted grouped results into heap form:
    {date: [(score, tiebreaker, payload_tuple), ...]}.
    """
    normalized_results = defaultdict(list)
    tiebreaker = 0

    for date, entries in raw_results.items():
        if not isinstance(entries, list):
            continue

        heap_entries = []
        for entry in entries:
            if not isinstance(entry, (tuple, list)) or len(entry) == 0:
                continue

            if _is_heap_result_item(entry):
                score = float(entry[0])
                payload = tuple(entry[2])
            else:
                score = float(entry[0])
                payload = tuple(entry)

            heap_entries.append((score, tiebreaker, payload))
            tiebreaker += 1

        if heap_entries:
            heapq.heapify(heap_entries)
        normalized_results[date] = heap_entries

    return normalized_results

def _heap_payload(entry: tuple | list) -> tuple:
    """Extracts a payload tuple from either heap or flattened result entry."""
    if _is_heap_result_item(entry):
        return tuple(entry[2])
    return tuple(entry)

def gen_weights(tickers: tuple[str, ...], increment: float = 0.05) -> Iterator[tuple[float, ...]]:
    """Generates portfolios using combinations of ticker weights."""

    # Check if more than 1 ticker was provided
    if len(tickers) == 1:
        yield (1.0,)
        return

    # Build permutations of main ticker weights
    rem_tickers = tickers[1:]
    num_rem = len(rem_tickers)

    # Scale of values - 100 for 1% increments, 1000 for 0.1% increments, etc..
    scale = 10000

    # Configure weight bounds and step sizes
    min_main_ticker_weight = int(0.0 * scale) # 0%
    max_main_ticker_weight = int(1.0 * scale) # 100%
    step_main_ticker_weight = int(round(increment * scale))

    min_rem_ticker_weight = int(0.00 * scale) # 0%
    step_rem_ticker_weight = int(round(increment * scale))
    if step_main_ticker_weight < 1 or step_rem_ticker_weight < 1:
        raise ValueError("Weight increment is too small for configured precision.")

    # Start with main ticker weights
    for main_ticker_weight in range(min_main_ticker_weight, max_main_ticker_weight + 1, step_main_ticker_weight):
        distributable_weight = scale - main_ticker_weight
        if distributable_weight == 0:
            yield tuple([1.0] + [0.0 for _ in range(len(rem_tickers))])
            continue

        # Recursively find valid combinations of remaining distributable weights
        def get_remaining_weights(remaining: int, count: int) -> Iterator[list[float]]:
            """Yields valid remaining-weight allocations that sum exactly."""
            if count == 1:
                if remaining >= min_rem_ticker_weight and remaining % step_rem_ticker_weight == 0:
                    yield [round(remaining / scale, 4)]
                return

            max_rem_ticker_weight = remaining - ((count - 1) * min_rem_ticker_weight)
            for weight in range(min_rem_ticker_weight, max_rem_ticker_weight + 1, step_rem_ticker_weight):
                for other_weights in get_remaining_weights(remaining - weight, count - 1):
                    yield [round(weight / scale, 4)] + other_weights

        final_main_weight = round(main_ticker_weight / scale, 4)
        for weights in get_remaining_weights(distributable_weight, num_rem):
            yield tuple([final_main_weight] + weights)

def gen_test_dates(
    price_history_dates: np.ndarray[np.datetime64],
    start_date: np.datetime64 | None = None,
    stop_date: np.datetime64 | None = None,
) -> Iterator[np.datetime64]:
    """Returns dates for backtesting."""

    # Sample size
    #sample_size = 1/3
    sample_size = 1/2

    if price_history_dates.size == 0:
        return

    # Bound requested dates to available trading dates.
    if start_date is None:
        bounded_start_date = np.datetime64(price_history_dates[0], "D")
    else:
        bounded_start_date = bound_to_available_dates(
            np.datetime64(start_date, "D"), price_history_dates, upper_bound=False
        )

    if stop_date is None:
        bounded_stop_date = np.datetime64(price_history_dates[-1], "D")
    else:
        bounded_stop_date = bound_to_available_dates(
            np.datetime64(stop_date, "D"), price_history_dates, upper_bound=True
        )

    if bounded_start_date > bounded_stop_date:
        return

    bounded_dates = price_history_dates[
        (price_history_dates >= bounded_start_date) & (price_history_dates <= bounded_stop_date)
    ]
    if bounded_dates.size == 0:
        return

    # Vectorized year/month extraction avoids repeated datetime conversions.
    years = bounded_dates.astype("datetime64[Y]").astype(int) + 1970
    months = (bounded_dates.astype("datetime64[M]").astype(int) % 12) + 1

    for year in range(int(years.min()), int(years.max()) + 1):
        year_mask = years == year
        if not np.any(year_mask):
            continue

        year_dates = bounded_dates[year_mask]
        year_months = months[year_mask]
        for month in range(1, 13):
            trading_days = year_dates[year_months == month]
            if trading_days.size == 0:
                continue
            yield from sample_strategy(trading_days, sample_ratio=sample_size)

def sample_strategy(
    population: list[np.datetime64] | np.ndarray,
    sample_ratio: float = 0.5,
    strategy: str = "random",
) -> Iterator[np.datetime64]:
    """Randomly selects trading days to simulate based on a specified sampling strategy."""
    sampling_strategy = strategy.lower()

    # Randomly sample from the population
    if sampling_strategy == "random":
        sample_size = int(len(population) * sample_ratio)
        if sample_size <= 0:
            return
        sample = np.random.choice(population, size=sample_size, replace=False)
        yield from sample
        #results.extend(sample)

    # Randomly pick from each day of the week, choosing an even distribution between days
    elif sampling_strategy == "specific_day_of_the_week":
        population = np.array(population, copy=True)
        np.random.shuffle(population)
        days_of_the_week = {
            1: [],
            2: [],
            3: [],
            4: [],
            5: []
        }
        for day in population:
            day_of_week = day.astype("datetime64[D]").item().isoweekday()
            if day_of_week in days_of_the_week:
                days_of_the_week[day_of_week].append(day)

        sample_size = len(min(days_of_the_week.values(), key = lambda x: len(x)))

        for days in days_of_the_week.values():
            yield from days[:sample_size]
            #results.extend(days[:sample_size])

def _iter_sim_parameters_from_index(sim_axes: dict[str, tuple], start_index: int = 0) -> Iterator[tuple]:
    """Yields deterministic Cartesian product cases from a starting global index."""
    weights = sim_axes["weights"]
    test_dates = sim_axes["test_dates"]
    allocation_schedules = sim_axes["allocation_schedules"]
    rebalance_schedules = sim_axes["rebalance_schedules"]

    weight_count = len(weights)
    date_count = len(test_dates)
    allocation_count = len(allocation_schedules)
    rebalance_count = len(rebalance_schedules)

    total = weight_count * date_count * allocation_count * rebalance_count
    if total == 0 or start_index >= total:
        return

    date_block = allocation_count * rebalance_count
    weight_block = date_count * date_block

    for index in range(start_index, total):
        weight_index = index // weight_block
        rem = index % weight_block
        date_index = rem // date_block
        rem = rem % date_block
        allocation_index = rem // rebalance_count
        rebalance_index = rem % rebalance_count

        yield (
            weights[weight_index],
            test_dates[date_index],
            allocation_schedules[allocation_index],
            rebalance_schedules[rebalance_index],
        )

def _build_simulation_cache_key(config: dict, price_dates: np.ndarray) -> str:
    """Builds a deterministic key for simulation cache/checkpoint artifacts."""
    weights_mode = config.get("weights_mode", "static")
    weights_signature = {
        "mode": weights_mode,
        "increment": config.get("weights_increment"),
    }
    allocation_mode = config.get("allocation_mode", "static")
    allocation_schedule = config.get("allocation_schedule")
    if allocation_mode == "dynamic":
        allocation_signature = {
            "mode": "dynamic",
            "increment": config.get("allocation_increment", 4),
        }
    else:
        allocation_signature = {
            "mode": "static",
            "schedule": (
                allocation_schedule.fmt.name,
                allocation_schedule.value,
            ) if allocation_schedule is not None else None,
        }
    rebalance_mode = config.get("rebalance_mode", "static")
    rebalance_schedule = config.get("rebalance_schedule")
    if rebalance_mode == "dynamic":
        rebalance_signature = {
            "mode": "dynamic",
            "increment": config.get("rebalance_increment", 4),
        }
    else:
        rebalance_signature = {
            "mode": "static",
            "schedule": (
                rebalance_schedule.fmt.name,
                rebalance_schedule.value,
            ) if rebalance_schedule is not None else None,
        }
    payload = {
        "securities": list(config.get("securities", ())),
        "weights": weights_signature,
        "dates": [str(config["dates"][0]), str(config["dates"][1])],
        "allocation_schedule": allocation_signature,
        "rebalance_schedule": rebalance_signature,
        "price_start": str(price_dates[0]) if price_dates.size else None,
        "price_stop": str(price_dates[-1]) if price_dates.size else None,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()[:12]

def get_sim_parameters(config: dict, price_dates: np.ndarray) -> tuple[dict[str, tuple], int]:
    """Generates or loads simulation axes used for deterministic parameter expansion."""
    simulation_key = _build_simulation_cache_key(config, price_dates)

    # Persist test parameters to file
    sim_parameters_file = os.path.join(os.getcwd(), f"simulation_parameters_{simulation_key}.pkl")
    if not os.path.exists(sim_parameters_file):
        tickers = tuple(security[0] for security in config.get("securities", ()))
        weights = (tuple(security[1] for security in config["securities"]),)
        allocation_schedules = (config.get("allocation_schedule"),)
        rebalance_schedules = (config.get("rebalance_schedule"),)

        # Generate portfolio weight permutations for simulation.
        weights_mode = config.get("weights_mode", "static")
        if weights_mode == "dynamic" and len(tickers) > 1:
            weight_increment = config.get("weights_increment", 0.05)
            weights = tuple(gen_weights(tickers, increment=weight_increment))

        allocation_mode = config.get("allocation_mode", "static")
        if allocation_mode == "dynamic":
            allocation_increment = config.get("allocation_increment", 4)
            allocation_schedules = tuple(
                Schedule(ScheduleFormat.DAYS, i) for i in range(1, 366, allocation_increment)
            )

        rebalance_mode = config.get("rebalance_mode", "static")
        if rebalance_mode == "dynamic":
            rebalance_increment = config.get("rebalance_increment", 4)
            rebalance_schedules = tuple(
                Schedule(ScheduleFormat.DAYS, i) for i in range(1, 366, rebalance_increment)
            )

        # Generate a list of possible test dates
        requested_start, requested_stop = config["dates"]
        start_date = coerce_date(requested_start, price_dates[0])
        stop_date = coerce_date(requested_stop, price_dates[-1])
        test_dates = tuple(gen_test_dates(price_dates, start_date=start_date, stop_date=stop_date))
        sim_axes = {
            "weights": weights,
            "test_dates": test_dates,
            "allocation_schedules": allocation_schedules,
            "rebalance_schedules": rebalance_schedules,
        }
        with open(sim_parameters_file, "wb") as f:
            pickle.dump(sim_axes, f)
    else:
        with open(sim_parameters_file, "rb") as f:
            sim_axes = pickle.load(f)

    total_cases = (
        len(sim_axes["weights"])
        * len(sim_axes["test_dates"])
        * len(sim_axes["allocation_schedules"])
        * len(sim_axes["rebalance_schedules"])
    )
    return sim_axes, total_cases

def _init_worker_shared_arrays(historical_data: tuple, dates: tuple) -> None:
    """Attaches each worker process to shared-memory arrays once."""
    global _WORKER_PRICE_DATA, _WORKER_DISTRIBUTION_DATA, _WORKER_PRICE_DATES, _WORKER_DISTRIBUTION_DATES
    global _WORKER_PRICE_SHM, _WORKER_DISTRIBUTION_SHM, _WORKER_PRICE_DATES_SHM, _WORKER_DISTRIBUTION_DATES_SHM

    (
        price_history_shm_name,
        price_history_shape,
        price_history_dtype,
        distribution_history_shm_name,
        distribution_history_shape,
        distribution_history_dtype,
    ) = historical_data
    (
        price_dates_shm_name,
        price_dates_shape,
        price_dates_dtype,
        distribution_dates_shm_name,
        distribution_dates_shape,
        distribution_dates_dtype,
    ) = dates

    _WORKER_PRICE_SHM = shared_memory.SharedMemory(name=price_history_shm_name)
    _WORKER_PRICE_DATES_SHM = shared_memory.SharedMemory(name=price_dates_shm_name)
    _WORKER_PRICE_DATA = np.ndarray(price_history_shape, dtype=price_history_dtype, buffer=_WORKER_PRICE_SHM.buf)
    _WORKER_PRICE_DATES = np.ndarray(price_dates_shape, dtype=price_dates_dtype, buffer=_WORKER_PRICE_DATES_SHM.buf)

    if distribution_history_shm_name is not None:
        _WORKER_DISTRIBUTION_SHM = shared_memory.SharedMemory(name=distribution_history_shm_name)
        _WORKER_DISTRIBUTION_DATA = np.ndarray(
            distribution_history_shape, dtype=distribution_history_dtype, buffer=_WORKER_DISTRIBUTION_SHM.buf
        )
    else:
        _WORKER_DISTRIBUTION_SHM = None
        _WORKER_DISTRIBUTION_DATA = None

    if distribution_dates_shm_name is not None:
        _WORKER_DISTRIBUTION_DATES_SHM = shared_memory.SharedMemory(name=distribution_dates_shm_name)
        _WORKER_DISTRIBUTION_DATES = np.ndarray(
            distribution_dates_shape, dtype=distribution_dates_dtype, buffer=_WORKER_DISTRIBUTION_DATES_SHM.buf
        )
    else:
        _WORKER_DISTRIBUTION_DATES_SHM = None
        _WORKER_DISTRIBUTION_DATES = None

def _test_portfolio(config: dict, test_case: tuple) -> tuple:
    """Tests a strategy on a single parameter combination."""
    weights, start_date, allocation_schedule, rebalance_schedule = test_case
    tickers = tuple(security[0] for security in config["securities"])

    portfolio = Portfolio(tickers, weights)
    strategy_args = dict(config.get("strategy_args", {}))
    reserved_strategy_kwargs = {
        "initial_allocation",
        "yearly_allocation",
        "allocation_schedule",
        "start_date",
        "track",
    }
    for key in reserved_strategy_kwargs:
        strategy_args.pop(key, None)
    if rebalance_schedule is not None:
        strategy_args["schedule"] = rebalance_schedule

    strategy = config["strategy"](
        initial_allocation=config["initial_allocation"],
        yearly_allocation=config["yearly_allocation"],
        allocation_schedule=allocation_schedule,
        start_date=start_date,
        track=config.get("track_performance", False),
        **strategy_args,
    )

    final_portfolio, parameters = strategy.execute(
        (_WORKER_PRICE_DATA, _WORKER_DISTRIBUTION_DATA),
        (_WORKER_PRICE_DATES, _WORKER_DISTRIBUTION_DATES),
        portfolio,
        trace=config.get("trace", False),
    )

    final_prices = _WORKER_PRICE_DATA[(_WORKER_PRICE_DATES == _WORKER_PRICE_DATES[-1]), 0]
    return (
        start_date,
        final_portfolio.current_value(final_prices),
        final_portfolio.total_new_capital,
        final_portfolio.total_distribution,
        final_portfolio.weights,
        *parameters.values(),
    )

def parallel(config: dict) -> None:
    """Parallel simulation of a trading strategy across multiple parameters."""

    # Get historical data
    try:
        price_data, distribution_data = get_historical_data((security[0] for security in config["securities"]))
    except RuntimeError as e:
        print(e)
        sys.exit(-1)

    # Extract histories and date ranges
    price_history_dates, price_history = price_data
    distribution_history_dates, distribution_history = distribution_data

    # Generate or load simulation parameter axes.
    sim_axes, total_simulations = get_sim_parameters(config, price_history_dates)
    simulation_key = _build_simulation_cache_key(config, price_history_dates)

    # Pickle results for persistence
    results_file = os.path.join(os.getcwd(), f"results_{simulation_key}.pkl")
    if os.path.exists(results_file):
        with open(results_file, 'rb') as f:
            results = _normalize_grouped_results_for_incremental(pickle.load(f))
    else:
        results = defaultdict(list)

    # Use a checkpoint file to periodically persist results to disk to account for interruptions
    checkpoint_file = os.path.join(os.getcwd(), f"checkpoint_{simulation_key}.txt")
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint = int(f.readline().replace("\n", ""))
    else:
        checkpoint = 0

    if checkpoint == -1 or checkpoint >= total_simulations:
        return results

    remaining_simulations = total_simulations - checkpoint
    sim_parameters = _iter_sim_parameters_from_index(sim_axes, checkpoint)

    # Profiling
    # historical_data = (price_history, distribution_history)
    # indices = (price_history_index, distribution_history_index)

    # test_function = partial(_test_portfolio, tickers, historical_data, indices)

    # test_cases = product(ticker_weight_permutations, strategy_permutations)
    # test_size = 100
    # start = time()
    # for i in range(test_size):
    #     test_case = next(test_cases)
    #     res = test_function(test_case)
    # stop = time()
    # run_time = stop - start
    # avg_run_time = run_time / test_size
    # print(f"{run_time:.3f} seconds total, {avg_run_time:.3f} seconds per task")
    # sys.exit()

    # Number of results to keep for each test case
    results_to_keep = 5

    # Multi-processing configuration parameters
    process_count = max(1, (os.cpu_count() or 1) - 1)
    max_chunk_size = 1500

    print("Creating shared memory blocks...")

    price_history_shm = None
    distribution_history_shm = None
    price_dates_shm = None
    distribution_dates_shm = None
    try:
        # Create shared memory for price history.
        price_history_shm = shared_memory.SharedMemory(create=True, size=price_history.nbytes)
        price_history_buf = np.ndarray(price_history.shape, dtype=price_history.dtype, buffer=price_history_shm.buf)
        price_history_buf[:] = price_history[:]

        # Create shared memory for price dates.
        price_dates_shm = shared_memory.SharedMemory(create=True, size=price_history_dates.nbytes)
        price_dates_buf = np.ndarray(
            price_history_dates.shape, dtype=price_history_dates.dtype, buffer=price_dates_shm.buf
        )
        price_dates_buf[:] = price_history_dates[:]

        distribution_meta = (None, None, None)
        distribution_dates_meta = (None, None, None)
        if distribution_history is not None and distribution_history_dates is not None:
            distribution_history_shm = shared_memory.SharedMemory(create=True, size=distribution_history.nbytes)
            distribution_history_buf = np.ndarray(
                distribution_history.shape, dtype=distribution_history.dtype, buffer=distribution_history_shm.buf
            )
            distribution_history_buf[:] = distribution_history[:]

            distribution_dates_shm = shared_memory.SharedMemory(create=True, size=distribution_history_dates.nbytes)
            distribution_dates_buf = np.ndarray(
                distribution_history_dates.shape,
                dtype=distribution_history_dates.dtype,
                buffer=distribution_dates_shm.buf,
            )
            distribution_dates_buf[:] = distribution_history_dates[:]

            distribution_meta = (
                distribution_history_shm.name,
                distribution_history.shape,
                distribution_history.dtype,
            )
            distribution_dates_meta = (
                distribution_dates_shm.name,
                distribution_history_dates.shape,
                distribution_history_dates.dtype,
            )

        # Organize historical data and dates.
        historical_data = (
            price_history_shm.name,
            price_history.shape,
            price_history.dtype,
            distribution_meta[0],
            distribution_meta[1],
            distribution_meta[2],
        )
        dates = (
            price_dates_shm.name,
            price_history_dates.shape,
            price_history_dates.dtype,
            distribution_dates_meta[0],
            distribution_dates_meta[1],
            distribution_dates_meta[2],
        )

        with Pool(
            processes=process_count,
            initializer=_init_worker_shared_arrays,
            initargs=(historical_data, dates),
        ) as worker_pool:

            print("Starting simulations...")

            # Create a partial function for shared parameters.
            test_function = partial(_test_portfolio, config)

            # Run all simulations.
            outcomes = worker_pool.imap(test_function, sim_parameters, max_chunk_size)

            # Keep the best results.
            checkpoint = top_n_grouped_incremental(
                results_to_keep,
                outcomes,
                remaining_simulations,
                results,
                checkpoint,
                checkpoint_file,
                results_file,
            )

            # Sort results.
            for date, heap in results.items():
                top_n_list = heapq.nlargest(results_to_keep, heap, key=lambda x: x[0])
                results[date] = [_heap_payload(item) for item in top_n_list]

            # Do a final write.
            with open(checkpoint_file, "w") as f:
                f.write(str(checkpoint))

            with open(results_file, "wb") as f:
                pickle.dump(results, f)
    finally:
        for shm in (
            price_history_shm,
            price_dates_shm,
            distribution_history_shm,
            distribution_dates_shm,
        ):
            if shm is None:
                continue
            shm.close()
            try:
                shm.unlink()
            except FileNotFoundError:
                pass

    return results