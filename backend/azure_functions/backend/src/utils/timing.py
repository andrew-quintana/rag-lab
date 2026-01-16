"""Timing utility functions"""

import time
from contextlib import contextmanager
from typing import Generator
from src.core.logging import get_logger

logger = get_logger("utils.timing")


@contextmanager
def timer(operation_name: str) -> Generator[None, None, None]:
    """
    Context manager for timing operations.
    
    Args:
        operation_name: Name of the operation being timed
        
    Example:
        with timer("chunking"):
            chunks = chunk_text(text)
    """
    start_time = time.time()
    logger.info(f"Starting {operation_name}")
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        logger.info(f"Completed {operation_name} in {elapsed:.2f}s")

