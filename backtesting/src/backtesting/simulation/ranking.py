"""Ranking and checkpoint utilities for simulation pipelines."""

import heapq
import pickle
from typing import Iterable

from tqdm import tqdm


def top_n_grouped_incremental(
    n: int, m: Iterable, size: int,
    results: dict, checkpoint: int,
    checkpoint_file: str, results_file: str
) -> int:
    """
    Returns the top N items grouped by np.datetime64.

    date_fn: Should return an np.datetime64 object.
    top_fn: Should return the numerical score for ranking.
    """
    checkpoint_target = 250000  # Persist after every 250,000 results
    last_date = None
    current_heap = None
    for item in tqdm(m, total=size):
        date, score = item[0], item[1]

        if date == last_date:
            heap = current_heap
        else:
            if date not in results:
                results[date] = []
            heap = results[date]
            current_heap = heap
            last_date = date

        heap_item = (score, checkpoint, item[1:])
        if len(heap) < n:
            heapq.heappush(heap, heap_item)
        elif score > heap[0][0]:
            heapq.heapreplace(heap, heap_item)

        checkpoint += 1
        if checkpoint % checkpoint_target == 0:
            with open(checkpoint_file, "w") as f:
                f.write(str(checkpoint))

            with open(results_file, "wb") as f:
                pickle.dump(results, f)

    return checkpoint
