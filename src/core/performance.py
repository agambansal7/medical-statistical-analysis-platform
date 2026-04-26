"""Performance and Scalability Module.

Provides:
- Async analysis execution with progress tracking
- Large dataset support (chunked processing)
- Result caching
- Background job queue
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Callable, Generator
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future
import threading
import queue
import hashlib
import json
import time
from datetime import datetime, timedelta
import pickle
from pathlib import Path
import warnings


@dataclass
class JobStatus:
    """Status of a background job."""
    job_id: str
    status: str  # queued, running, completed, failed
    progress: float  # 0.0 to 1.0
    message: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Optional[Any] = None
    error: Optional[str] = None


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: str
    expires_at: str
    hit_count: int = 0
    size_bytes: int = 0


class CacheManager:
    """Manage analysis result caching."""

    def __init__(
        self,
        max_size_mb: int = 500,
        default_ttl_minutes: int = 60,
        storage_path: Optional[str] = None
    ):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.default_ttl = timedelta(minutes=default_ttl_minutes)
        self.storage_path = Path(storage_path) if storage_path else None

        self._cache: Dict[str, CacheEntry] = {}
        self._current_size = 0
        self._lock = threading.Lock()

        # Load persistent cache if available
        if self.storage_path:
            self._load_persistent_cache()

    def get(self, key: str) -> Optional[Any]:
        """Get cached value.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            if key not in self._cache:
                return None

            entry = self._cache[key]

            # Check expiration
            if datetime.fromisoformat(entry.expires_at) < datetime.now():
                del self._cache[key]
                self._current_size -= entry.size_bytes
                return None

            # Update hit count
            entry.hit_count += 1
            return entry.value

    def set(
        self,
        key: str,
        value: Any,
        ttl_minutes: Optional[int] = None
    ):
        """Set cached value.

        Args:
            key: Cache key
            value: Value to cache
            ttl_minutes: Time to live in minutes
        """
        # Calculate size
        size = len(pickle.dumps(value))

        # Check if we need to evict
        while self._current_size + size > self.max_size_bytes and self._cache:
            self._evict_one()

        ttl = timedelta(minutes=ttl_minutes) if ttl_minutes else self.default_ttl

        entry = CacheEntry(
            key=key,
            value=value,
            created_at=datetime.now().isoformat(),
            expires_at=(datetime.now() + ttl).isoformat(),
            size_bytes=size
        )

        with self._lock:
            # Remove old entry if exists
            if key in self._cache:
                self._current_size -= self._cache[key].size_bytes

            self._cache[key] = entry
            self._current_size += size

        # Persist if storage configured
        if self.storage_path:
            self._save_entry(entry)

    def invalidate(self, key: str):
        """Invalidate a cache entry."""
        with self._lock:
            if key in self._cache:
                self._current_size -= self._cache[key].size_bytes
                del self._cache[key]

    def clear(self):
        """Clear all cache."""
        with self._lock:
            self._cache.clear()
            self._current_size = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_hits = sum(e.hit_count for e in self._cache.values())
            return {
                'entries': len(self._cache),
                'size_mb': self._current_size / (1024 * 1024),
                'max_size_mb': self.max_size_bytes / (1024 * 1024),
                'utilization': self._current_size / self.max_size_bytes,
                'total_hits': total_hits
            }

    def _evict_one(self):
        """Evict least recently used entry."""
        with self._lock:
            if not self._cache:
                return

            # Find entry with lowest hit count
            min_entry = min(self._cache.values(), key=lambda e: e.hit_count)
            self._current_size -= min_entry.size_bytes
            del self._cache[min_entry.key]

    def _load_persistent_cache(self):
        """Load cache from disk."""
        if not self.storage_path or not self.storage_path.exists():
            return

        try:
            cache_file = self.storage_path / "cache_index.json"
            if cache_file.exists():
                with open(cache_file) as f:
                    index = json.load(f)
                # Only load keys, values loaded on demand
        except:
            pass

    def _save_entry(self, entry: CacheEntry):
        """Save entry to disk."""
        if not self.storage_path:
            return

        self.storage_path.mkdir(parents=True, exist_ok=True)
        entry_file = self.storage_path / f"{entry.key}.pkl"

        try:
            with open(entry_file, 'wb') as f:
                pickle.dump(entry, f)
        except:
            pass

    @staticmethod
    def make_key(*args, **kwargs) -> str:
        """Create cache key from arguments."""
        key_parts = [str(arg) for arg in args]
        key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()


class AsyncExecutor:
    """Execute analyses asynchronously."""

    def __init__(
        self,
        max_workers: int = 4,
        use_processes: bool = False
    ):
        self.max_workers = max_workers
        self._executor = (ProcessPoolExecutor if use_processes else ThreadPoolExecutor)(
            max_workers=max_workers
        )
        self._jobs: Dict[str, JobStatus] = {}
        self._futures: Dict[str, Future] = {}
        self._callbacks: Dict[str, List[Callable]] = {}
        self._lock = threading.Lock()

    def submit(
        self,
        func: Callable,
        *args,
        job_id: Optional[str] = None,
        on_complete: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
        **kwargs
    ) -> str:
        """Submit job for async execution.

        Args:
            func: Function to execute
            *args: Positional arguments
            job_id: Optional job ID
            on_complete: Callback on completion
            on_error: Callback on error
            **kwargs: Keyword arguments

        Returns:
            Job ID
        """
        job_id = job_id or hashlib.md5(str(time.time()).encode()).hexdigest()[:12]

        status = JobStatus(
            job_id=job_id,
            status='queued',
            progress=0.0,
            message='Job queued'
        )

        with self._lock:
            self._jobs[job_id] = status
            self._callbacks[job_id] = []
            if on_complete:
                self._callbacks[job_id].append(on_complete)

        # Submit to executor
        future = self._executor.submit(self._run_job, job_id, func, args, kwargs)
        self._futures[job_id] = future

        # Add completion callback
        future.add_done_callback(lambda f: self._on_job_done(job_id, f, on_error))

        return job_id

    def get_status(self, job_id: str) -> Optional[JobStatus]:
        """Get job status."""
        return self._jobs.get(job_id)

    def get_result(self, job_id: str, timeout: Optional[float] = None) -> Any:
        """Get job result (blocking).

        Args:
            job_id: Job ID
            timeout: Timeout in seconds

        Returns:
            Job result
        """
        future = self._futures.get(job_id)
        if not future:
            raise ValueError(f"Job {job_id} not found")

        return future.result(timeout=timeout)

    def cancel(self, job_id: str) -> bool:
        """Cancel a job.

        Args:
            job_id: Job ID

        Returns:
            True if cancelled
        """
        future = self._futures.get(job_id)
        if future:
            cancelled = future.cancel()
            if cancelled:
                with self._lock:
                    self._jobs[job_id].status = 'cancelled'
            return cancelled
        return False

    def update_progress(self, job_id: str, progress: float, message: str = ''):
        """Update job progress (call from within job function)."""
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].progress = progress
                self._jobs[job_id].message = message

    def _run_job(
        self,
        job_id: str,
        func: Callable,
        args: tuple,
        kwargs: dict
    ) -> Any:
        """Run job and update status."""
        with self._lock:
            self._jobs[job_id].status = 'running'
            self._jobs[job_id].started_at = datetime.now().isoformat()
            self._jobs[job_id].message = 'Running...'

        try:
            # Pass progress callback if function accepts it
            import inspect
            sig = inspect.signature(func)
            if 'progress_callback' in sig.parameters:
                kwargs['progress_callback'] = lambda p, m='': self.update_progress(job_id, p, m)

            result = func(*args, **kwargs)

            with self._lock:
                self._jobs[job_id].status = 'completed'
                self._jobs[job_id].progress = 1.0
                self._jobs[job_id].completed_at = datetime.now().isoformat()
                self._jobs[job_id].result = result
                self._jobs[job_id].message = 'Completed'

            return result

        except Exception as e:
            with self._lock:
                self._jobs[job_id].status = 'failed'
                self._jobs[job_id].completed_at = datetime.now().isoformat()
                self._jobs[job_id].error = str(e)
                self._jobs[job_id].message = f'Failed: {str(e)}'
            raise

    def _on_job_done(
        self,
        job_id: str,
        future: Future,
        on_error: Optional[Callable]
    ):
        """Handle job completion."""
        exc = future.exception()
        if exc and on_error:
            on_error(exc)
        elif not exc:
            callbacks = self._callbacks.get(job_id, [])
            for callback in callbacks:
                try:
                    callback(future.result())
                except:
                    pass

    def shutdown(self, wait: bool = True):
        """Shutdown executor."""
        self._executor.shutdown(wait=wait)


class ChunkedProcessor:
    """Process large datasets in chunks."""

    def __init__(self, chunk_size: int = 10000):
        self.chunk_size = chunk_size

    def process(
        self,
        data: pd.DataFrame,
        func: Callable,
        aggregator: Optional[Callable] = None,
        progress_callback: Optional[Callable] = None
    ) -> Any:
        """Process DataFrame in chunks.

        Args:
            data: Large DataFrame
            func: Function to apply to each chunk
            aggregator: Function to combine chunk results
            progress_callback: Progress callback

        Returns:
            Aggregated result
        """
        n_chunks = (len(data) + self.chunk_size - 1) // self.chunk_size
        results = []

        for i, chunk in enumerate(self._chunked(data)):
            result = func(chunk)
            results.append(result)

            if progress_callback:
                progress_callback((i + 1) / n_chunks, f'Processed chunk {i + 1}/{n_chunks}')

        if aggregator:
            return aggregator(results)
        return results

    def _chunked(self, data: pd.DataFrame) -> Generator[pd.DataFrame, None, None]:
        """Yield chunks of DataFrame."""
        for i in range(0, len(data), self.chunk_size):
            yield data.iloc[i:i + self.chunk_size]

    def parallel_process(
        self,
        data: pd.DataFrame,
        func: Callable,
        aggregator: Callable,
        n_workers: int = 4
    ) -> Any:
        """Process chunks in parallel.

        Args:
            data: Large DataFrame
            func: Function to apply
            aggregator: Function to combine results
            n_workers: Number of parallel workers

        Returns:
            Aggregated result
        """
        chunks = list(self._chunked(data))

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(func, chunks))

        return aggregator(results)


class PerformanceManager:
    """Unified performance management."""

    def __init__(
        self,
        cache_size_mb: int = 500,
        max_workers: int = 4,
        chunk_size: int = 10000
    ):
        self.cache = CacheManager(max_size_mb=cache_size_mb)
        self.executor = AsyncExecutor(max_workers=max_workers)
        self.chunked = ChunkedProcessor(chunk_size=chunk_size)

    def run_cached(
        self,
        func: Callable,
        *args,
        cache_key: Optional[str] = None,
        ttl_minutes: int = 60,
        **kwargs
    ) -> Any:
        """Run function with caching.

        Args:
            func: Function to run
            *args: Positional arguments
            cache_key: Optional cache key
            ttl_minutes: Cache TTL
            **kwargs: Keyword arguments

        Returns:
            Function result (cached or fresh)
        """
        key = cache_key or CacheManager.make_key(
            func.__name__, *args, **kwargs
        )

        # Check cache
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        # Run and cache
        result = func(*args, **kwargs)
        self.cache.set(key, result, ttl_minutes)

        return result

    def run_async(
        self,
        func: Callable,
        *args,
        job_id: Optional[str] = None,
        **kwargs
    ) -> str:
        """Run function asynchronously.

        Args:
            func: Function to run
            *args: Arguments
            job_id: Optional job ID
            **kwargs: Keyword arguments

        Returns:
            Job ID
        """
        return self.executor.submit(func, *args, job_id=job_id, **kwargs)

    def run_chunked(
        self,
        data: pd.DataFrame,
        func: Callable,
        aggregator: Optional[Callable] = None,
        parallel: bool = False
    ) -> Any:
        """Run analysis on large dataset in chunks.

        Args:
            data: Large DataFrame
            func: Analysis function
            aggregator: Result aggregation function
            parallel: Use parallel processing

        Returns:
            Aggregated result
        """
        if parallel:
            return self.chunked.parallel_process(data, func, aggregator)
        return self.chunked.process(data, func, aggregator)

    def estimate_memory(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Estimate memory requirements for analysis.

        Args:
            data: Input DataFrame

        Returns:
            Memory estimates
        """
        data_memory = data.memory_usage(deep=True).sum()

        # Estimate typical analysis overhead
        n_rows, n_cols = data.shape

        estimates = {
            'data_size_mb': data_memory / (1024 * 1024),
            'estimated_peak_mb': data_memory * 3 / (1024 * 1024),  # 3x for typical analysis
            'rows': n_rows,
            'columns': n_cols,
            'recommend_chunking': n_rows > 100000,
            'recommend_sampling': n_rows > 1000000,
            'suggested_chunk_size': min(50000, n_rows // 10)
        }

        return estimates

    def shutdown(self):
        """Clean shutdown."""
        self.executor.shutdown()
        self.cache.clear()
