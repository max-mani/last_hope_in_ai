"""
Shared bounded background task pool.

Several parts of UYIR (Firebase upload, incident clip extraction, LLM
analysis) fire off background work whenever an incident is confirmed.
Previously each of these spawned a raw, uncapped `threading.Thread`.
Under a burst of near-simultaneous incidents — a multi-vehicle pileup,
or a mis-tuned camera producing rapid false triggers — that could spin
up an unbounded number of concurrent threads doing ffmpeg transcodes
and network I/O at once. This module centralizes that work behind one
bounded ThreadPoolExecutor shared by every caller (firebase_uploader.py,
app.py, stream_processor.py), sized by config.MAX_BACKGROUND_WORKERS.
"""

import concurrent.futures

import config

_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=max(1, config.MAX_BACKGROUND_WORKERS),
    thread_name_prefix="uyir-bg",
)


def submit(fn, *args, **kwargs):
    """Submit background work to the shared pool. Returns a concurrent.futures.Future."""
    return _executor.submit(fn, *args, **kwargs)


def shutdown(wait=True):
    _executor.shutdown(wait=wait)
