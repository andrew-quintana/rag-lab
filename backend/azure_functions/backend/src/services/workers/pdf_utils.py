"""PDF utilities for batch processing

This module provides utilities for PDF page counting and slicing
to support batched document processing in the ingestion worker.
"""

from typing import List, Tuple
from io import BytesIO
import PyPDF2
from src.core.exceptions import ValidationError
from src.core.logging import get_logger

logger = get_logger("services.workers.pdf_utils")


def get_pdf_page_count(pdf_bytes: bytes) -> int:
    """
    Get the total number of pages in a PDF.
    
    Uses PyPDF2 to count pages. Pages are 1-indexed for Azure Document Intelligence
    compatibility (PyPDF2 uses 0-indexed internally).
    
    Args:
        pdf_bytes: Raw PDF file content (bytes)
        
    Returns:
        Total number of pages (1-indexed count)
        
    Raises:
        ValueError: If PDF is invalid, corrupted, or empty
        ValidationError: If PDF cannot be read
    """
    if not pdf_bytes or len(pdf_bytes) == 0:
        raise ValueError("PDF bytes cannot be empty")
    
    try:
        pdf_file = BytesIO(pdf_bytes)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        # PyPDF2 uses 0-indexed pages, but we return 1-indexed count for Azure compatibility
        num_pages = len(pdf_reader.pages)
        
        if num_pages == 0:
            raise ValueError("PDF has no pages")
        
        logger.debug(f"Detected {num_pages} pages in PDF")
        return num_pages
        
    except PyPDF2.errors.PdfReadError as e:
        logger.error(f"Failed to read PDF: {e}")
        raise ValidationError(f"Invalid or corrupted PDF: {str(e)}") from e
    except Exception as e:
        logger.error(f"Unexpected error counting PDF pages: {e}")
        raise ValidationError(f"Failed to count PDF pages: {str(e)}") from e


def slice_pdf_to_batch(original_pdf_bytes: bytes, start_page: int, end_page: int) -> bytes:
    """
    Slice a PDF to extract a specific page range.
    
    This function extracts pages [start_page, end_page) from the original PDF
    and creates a new in-memory PDF containing only those pages.
    
    Page numbers are 1-indexed (Azure Document Intelligence format):
    - start_page: First page to include (1-indexed, inclusive)
    - end_page: Last page to include (1-indexed, exclusive)
    - Example: start_page=1, end_page=3 extracts pages 1 and 2
    
    Args:
        original_pdf_bytes: Raw PDF file content (bytes)
        start_page: First page to extract (1-indexed, inclusive)
        end_page: Last page to extract (1-indexed, exclusive)
        
    Returns:
        Bytes of new PDF containing only the specified page range
        
    Raises:
        ValueError: If page range is invalid or PDF is corrupted
        ValidationError: If PDF cannot be sliced
    """
    if not original_pdf_bytes or len(original_pdf_bytes) == 0:
        raise ValueError("PDF bytes cannot be empty")
    
    if start_page < 1:
        raise ValueError(f"start_page must be >= 1, got {start_page}")
    
    if end_page <= start_page:
        raise ValueError(f"end_page ({end_page}) must be > start_page ({start_page})")
    
    try:
        # Read original PDF
        pdf_file = BytesIO(original_pdf_bytes)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        total_pages = len(pdf_reader.pages)
        
        # Convert 1-indexed to 0-indexed for PyPDF2
        start_idx = start_page - 1
        end_idx = end_page - 1
        
        # Validate page range
        if start_idx >= total_pages:
            raise ValueError(
                f"start_page ({start_page}) exceeds total pages ({total_pages})"
            )
        
        if end_idx > total_pages:
            logger.warning(
                f"end_page ({end_page}) exceeds total pages ({total_pages}), "
                f"adjusting to {total_pages}"
            )
            end_idx = total_pages
        
        # Create new PDF with selected pages
        pdf_writer = PyPDF2.PdfWriter()
        
        for page_num in range(start_idx, end_idx):
            page = pdf_reader.pages[page_num]
            pdf_writer.add_page(page)
        
        # Write to bytes
        output_buffer = BytesIO()
        pdf_writer.write(output_buffer)
        output_buffer.seek(0)
        sliced_pdf_bytes = output_buffer.read()
        
        logger.debug(
            f"Sliced PDF: pages {start_page}-{end_idx} "
            f"(extracted {end_idx - start_idx} pages, "
            f"{len(sliced_pdf_bytes)} bytes)"
        )
        
        return sliced_pdf_bytes
        
    except PyPDF2.errors.PdfReadError as e:
        logger.error(f"Failed to read PDF for slicing: {e}")
        raise ValidationError(f"Invalid or corrupted PDF: {str(e)}") from e
    except Exception as e:
        logger.error(f"Unexpected error slicing PDF: {e}")
        raise ValidationError(f"Failed to slice PDF: {str(e)}") from e


def generate_page_batches(num_pages: int, batch_size: int = 2) -> List[Tuple[int, int]]:
    """
    Generate page batch ranges for batch processing.
    
    Creates a list of (start_page, end_page) tuples representing page ranges
    to process in batches. Pages are 1-indexed (Azure Document Intelligence format).
    
    Args:
        num_pages: Total number of pages in the PDF (must be > 0)
        batch_size: Number of pages per batch (default: 2, must be > 0)
        
    Returns:
        List of (start_page, end_page) tuples where:
        - start_page: First page in batch (1-indexed, inclusive)
        - end_page: Last page in batch (1-indexed, exclusive)
        - Example for 10 pages, batch_size=2: [(1, 3), (3, 5), (5, 7), (7, 9), (9, 11)]
        
    Raises:
        ValueError: If num_pages <= 0 or batch_size <= 0
    """
    if num_pages <= 0:
        raise ValueError(f"num_pages must be > 0, got {num_pages}")
    
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    
    batches = []
    start_page = 1  # 1-indexed
    
    while start_page <= num_pages:
        end_page = min(start_page + batch_size, num_pages + 1)  # Exclusive end
        batches.append((start_page, end_page))
        start_page = end_page
    
    logger.debug(
        f"Generated {len(batches)} batches for {num_pages} pages "
        f"(batch_size={batch_size})"
    )
    
    return batches

