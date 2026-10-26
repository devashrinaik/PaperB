"""GPU memory checks and OOM-retry with shrinking batch."""
import logging
import time

import torch

log = logging.getLogger(__name__)


def wait_for_free_memory(device, need_gb, poll_s=60, max_wait_s=6 * 3600):
    """Block until `need_gb` GiB are free on `device` (shared cards)."""
    t0 = time.time()
    while True:
        free, total = torch.cuda.mem_get_info(device)
        free_gb = free / 2**30
        if free_gb >= need_gb:
            log.info("cuda:%s free %.1f / %.1f GiB (need %.1f)", device, free_gb, total / 2**30, need_gb)
            return free_gb
        if time.time() - t0 > max_wait_s:
            raise RuntimeError(f"cuda:{device}: only {free_gb:.1f} GiB free after {max_wait_s}s, need {need_gb}")
        log.warning("cuda:%s only %.1f GiB free, need %.1f; waiting %ds", device, free_gb, need_gb, poll_s)
        time.sleep(poll_s)


def run_batched(fn, items, batch_size, min_batch=1):
    """Apply fn to consecutive chunks of `items`, halving the batch on OOM.

    fn(list) -> list of per-item results. Returns (results, final_batch_size).
    """
    out, i, bs = [], 0, batch_size
    while i < len(items):
        chunk = items[i:i + bs]
        try:
            out.extend(fn(chunk))
            i += len(chunk)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs <= min_batch:
                raise
            bs = max(min_batch, bs // 2)
            log.warning("OOM; retrying with batch %d", bs)
    return out, bs
