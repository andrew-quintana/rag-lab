"""Document ingestion using Azure Document Intelligence"""

from typing import Optional
import io
import math
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from rag_eval.core.interfaces import Chunk
from rag_eval.core.exceptions import AzureServiceError
from rag_eval.core.logging import get_logger

logger = get_logger("services.rag.ingestion")


def _extract_page_range(client: DocumentIntelligenceClient, file_content: bytes, page_range: str) -> tuple[list[str], int]:
    """
    Extract text from a specific page range.
    
    Args:
        client: Document Intelligence client
        file_content: Raw file content (bytes)
        page_range: Page range string (e.g., "1-2", "3-4")
        
    Returns:
        Tuple of (list of page texts, number of pages detected)
    """
    from io import BytesIO
    
    poller = client.begin_analyze_document(
        model_id="prebuilt-read",
        body=BytesIO(file_content),
        pages=page_range
    )
    
    result = poller.result(timeout=600)
    
    page_texts = []
    if hasattr(result, 'pages') and result.pages:
        for page in result.pages:
            page_lines = []
            
            # Extract from lines
            if hasattr(page, 'lines') and page.lines:
                for line in page.lines:
                    if hasattr(line, 'content') and line.content:
                        page_lines.append(line.content)
            
            # Fallback to paragraphs
            if not page_lines and hasattr(page, 'paragraphs') and page.paragraphs:
                for para in page.paragraphs:
                    if hasattr(para, 'content') and para.content:
                        page_lines.append(para.content)
            
            # Fallback to words
            if not page_lines and hasattr(page, 'words') and page.words:
                words = [w.content for w in page.words if hasattr(w, 'content') and w.content]
                if words:
                    page_lines.append(" ".join(words))
            
            if page_lines:
                page_texts.append("\n".join(page_lines))
    
    # Also get content field if available
    if hasattr(result, 'content') and result.content and result.content.strip():
        # Use content field if it's longer than page-based extraction
        content_text = result.content
        if len(content_text) > sum(len(pt) for pt in page_texts):
            page_texts = [content_text]
    
    pages_detected = len(result.pages) if hasattr(result, 'pages') and result.pages else 0
    return page_texts, pages_detected


def _get_total_pages_from_azure(client: DocumentIntelligenceClient, file_content: bytes) -> Optional[int]:
    """
    Try to get total page count from Azure by requesting a large page range.
    Azure will return the actual number of pages available.
    
    Args:
        client: Document Intelligence client
        file_content: Raw file content (bytes)
        
    Returns:
        Total page count if detected, None otherwise
    """
    from io import BytesIO
    
    try:
        # Request a very large page range - Azure will process what's available
        poller = client.begin_analyze_document(
            model_id="prebuilt-read",
            body=BytesIO(file_content),
            pages="1-10000"  # Request way more than any document would have
        )
        result = poller.result(timeout=600)
        
        # Check if result has page count metadata
        if hasattr(result, 'pages') and result.pages:
            # Try to infer total pages from the last page number
            # Note: This is a heuristic - Azure might not return all pages
            # but we can use the highest page number we see
            max_page_num = len(result.pages)
            
            # Also check if there's a totalPages attribute (if available in future SDK versions)
            if hasattr(result, 'total_pages'):
                return result.total_pages
            
            return max_page_num
    except Exception as e:
        logger.debug(f"Could not determine total pages from Azure: {e}")
    
    return None


def _batch_extract_pages(client: DocumentIntelligenceClient, file_content: bytes, estimated_pages: int) -> str:
    """
    Batch extract pages by processing 2 pages at a time (Free tier limit).
    Automatically detects when we've reached the end of the document.
    
    Args:
        client: Document Intelligence client
        file_content: Raw file content (bytes)
        estimated_pages: Estimated number of pages (will auto-detect end)
        
    Returns:
        Combined extracted text from all pages
    """
    logger.info(f"Batch processing up to {estimated_pages} pages in 2-page increments (Free tier workaround)")
    
    all_page_texts = []
    pages_processed = 0
    consecutive_empty = 0
    max_consecutive_empty = 3  # Stop after 3 consecutive empty batches
    
    # Process in 2-page batches
    for start_page in range(1, estimated_pages + 1, 2):
        end_page = min(start_page + 1, estimated_pages)
        page_range = f"{start_page}-{end_page}"
        
        logger.debug(f"Processing pages {page_range}...")
        try:
            page_texts, pages_detected = _extract_page_range(client, file_content, page_range)
            
            if page_texts and any(len(pt.strip()) > 0 for pt in page_texts):
                # Got text - reset empty counter
                consecutive_empty = 0
                all_page_texts.extend(page_texts)
                pages_processed += pages_detected
                chars_extracted = sum(len(pt) for pt in page_texts)
                logger.debug(f"  Extracted {chars_extracted:,} chars from {pages_detected} pages")
            else:
                # No text extracted - increment empty counter
                consecutive_empty += 1
                logger.debug(f"  No text extracted from pages {page_range} (empty batch {consecutive_empty}/{max_consecutive_empty})")
                
                # If we've hit multiple consecutive empty batches, we've likely reached the end
                if consecutive_empty >= max_consecutive_empty:
                    logger.info(f"Reached end of document after {pages_processed} pages (3 consecutive empty batches)")
                    break
                    
        except Exception as e:
            logger.warning(f"Failed to extract pages {page_range}: {e}. Continuing...")
            consecutive_empty += 1
            if consecutive_empty >= max_consecutive_empty:
                break
            continue
    
    logger.info(f"Batch processing complete: {pages_processed} pages processed, {len(all_page_texts)} page texts extracted")
    return "\n\n".join(all_page_texts)


def extract_text_from_document(file_content: bytes, config) -> str:
    """
    Extract text from a document using Azure Document Intelligence.
    Handles OCR, table extraction, and text segmentation.
    
    This function uses Azure Document Intelligence's "prebuilt-read" model to extract
    text from various document formats (PDF, images, etc.). It extracts:
    - Main document text content
    - Table cell contents (if tables are present)
    
    The extracted text is deterministic for the same input document.
    
    Args:
        file_content: Raw file content (bytes). Supports PDF, images, and other formats
                     supported by Azure Document Intelligence.
        config: Application configuration with Azure Document Intelligence credentials
               (azure_document_intelligence_endpoint, azure_document_intelligence_api_key)
        
    Returns:
        Extracted plaintext content. Text from main content and tables are joined with
        double newlines ("\\n\\n").
        
    Raises:
        AzureServiceError: If extraction fails (network errors, authentication failures,
                          invalid document format, service unavailability)
        
    Note:
        - Requires valid Azure Document Intelligence credentials in config
        - Large documents may take significant time to process
        - Empty documents will return empty string
        - Table extraction preserves cell content but not table structure
    """
    logger.info(f"Extracting text using Azure Document Intelligence (document size: {len(file_content):,} bytes)")
    
    try:
        # Initialize Document Intelligence client
        client = DocumentIntelligenceClient(
            endpoint=config.azure_document_intelligence_endpoint,
            credential=AzureKeyCredential(config.azure_document_intelligence_api_key)
        )
        
        # Analyze document (supports PDF, images, etc.)
        # This performs OCR, table extraction, and text segmentation
        # Wrap bytes in BytesIO for the API
        from io import BytesIO
        
        # Use direct body parameter - AnalyzeDocumentRequest doesn't work with this SDK version
        # pages=None means process all pages (required for multi-page documents)
        # Without explicit pages parameter, service may default to first 2 pages on Free tier
        poller = client.begin_analyze_document(
            model_id="prebuilt-read",  # Prebuilt model for text extraction
            body=BytesIO(file_content),
            pages=None  # Explicitly set to None to process all pages
        )
        
        # Set timeout for large documents (10 minutes)
        # Large documents (200+ pages) may take several minutes to process
        result = poller.result(timeout=600)
        
        # Extract text from all pages
        extracted_text = []
        
        # For scanned/image-based PDFs, extract text from pages/paragraphs (better for OCR)
        # This method often yields more text than the content field for scanned documents
        page_texts = []
        pages_detected = 0
        if hasattr(result, 'pages') and result.pages:
            pages_detected = len(result.pages)
            logger.info(f"Found {pages_detected} pages, extracting text from page structure")
            for page in result.pages:
                page_lines = []
                
                # Extract from lines (most common structure)
                if hasattr(page, 'lines') and page.lines:
                    for line in page.lines:
                        if hasattr(line, 'content') and line.content:
                            page_lines.append(line.content)
                
                # Fallback to paragraphs if lines not available
                if not page_lines and hasattr(page, 'paragraphs') and page.paragraphs:
                    for para in page.paragraphs:
                        if hasattr(para, 'content') and para.content:
                            page_lines.append(para.content)
                
                # Fallback to words if neither lines nor paragraphs available
                if not page_lines and hasattr(page, 'words') and page.words:
                    words = [w.content for w in page.words if hasattr(w, 'content') and w.content]
                    if words:
                        page_lines.append(" ".join(words))
                
                if page_lines:
                    page_texts.append("\n".join(page_lines))
        
        # Use page-based extraction if available, otherwise fall back to content field
        if page_texts:
            page_text = "\n\n".join(page_texts)
            extracted_text.append(page_text)
            logger.info(f"Extracted {len(page_text)} characters from {len(page_texts)} pages")
        elif result.content:
            # Fallback to content field for text-based PDFs
            extracted_text.append(result.content)
            logger.info(f"Extracted {len(result.content)} characters from content field")
        
        # Also extract text from tables if present
        if result.tables:
            for table in result.tables:
                for cell in table.cells:
                    if cell.content:
                        extracted_text.append(cell.content)
        
        full_text = "\n\n".join(extracted_text)
        text_length = len(full_text)
        logger.info(f"Extracted {text_length} characters from document")
        
        # Check if we hit the Free tier 2-page limit and need batch processing
        # Heuristic: If we got exactly 2 pages and the document is large, try batch processing
        if pages_detected == 2 and len(file_content) > 1_000_000:  # Files > 1MB
            expected_min_text = len(file_content) * 0.0001  # 0.01% of file size
            if text_length < expected_min_text and text_length < 2000:
                logger.warning(
                    f"Only 2 pages detected with minimal text ({text_length} chars) for large file "
                    f"({len(file_content):,} bytes). This suggests Free (F0) tier 2-page-per-request limit. "
                    f"Attempting batch processing to extract all pages..."
                )
                
                # Try to get actual page count from Azure first
                total_pages = _get_total_pages_from_azure(client, file_content)
                
                if total_pages and total_pages > 2:
                    logger.info(f"Detected {total_pages} total pages from Azure")
                    estimated_pages = total_pages
                else:
                    # Estimate total pages (rough heuristic: ~10KB per page for scanned PDFs)
                    estimated_pages = max(10, math.ceil(len(file_content) / 10_000))
                    logger.info(f"Could not detect page count from Azure. Estimating {estimated_pages} pages based on file size")
                
                # Try batch processing with detected/estimated page count
                # We'll process in batches and stop when we get empty results
                try:
                    batch_text = _batch_extract_pages(client, file_content, estimated_pages)
                    if len(batch_text) > text_length * 2:  # Only use if significantly more text
                        logger.info(
                            f"Batch processing successful: extracted {len(batch_text):,} characters "
                            f"(vs {text_length:,} from single request)"
                        )
                        full_text = batch_text
                        text_length = len(full_text)
                    else:
                        logger.info("Batch processing didn't yield significantly more text, using original result")
                except Exception as e:
                    logger.warning(f"Batch processing failed: {e}. Using original 2-page extraction.")
        
        logger.info(f"Final extraction: {text_length:,} characters")
        return full_text
        
    except Exception as e:
        # Enhanced error handling with specific error types
        from azure.core.exceptions import (
            HttpResponseError,
            ResourceNotFoundError,
            ClientAuthenticationError,
            ServiceRequestError
        )
        
        if isinstance(e, ClientAuthenticationError):
            logger.error(f"Azure Document Intelligence authentication failed: {e}")
            raise AzureServiceError(f"Authentication failed. Check your API key and endpoint: {str(e)}")
        elif isinstance(e, ResourceNotFoundError):
            logger.error(f"Azure Document Intelligence resource not found: {e}")
            raise AzureServiceError(f"Resource not found. Check your endpoint URL: {str(e)}")
        elif isinstance(e, HttpResponseError):
            if e.status_code == 429:
                logger.error("Azure Document Intelligence rate limit exceeded")
                raise AzureServiceError("Rate limit exceeded. Please retry later or upgrade your tier.")
            elif e.status_code == 413:
                logger.error("Document too large for Azure Document Intelligence")
                raise AzureServiceError(f"Document too large. Maximum size: 500 MB for Standard tier.")
            else:
                logger.error(f"Azure Document Intelligence HTTP error ({e.status_code}): {e}")
                raise AzureServiceError(f"HTTP error ({e.status_code}): {str(e)}")
        elif isinstance(e, ServiceRequestError):
            logger.error(f"Azure Document Intelligence service request failed: {e}")
            raise AzureServiceError(f"Service request failed. Check network connectivity: {str(e)}")
        else:
            logger.error(f"Document Intelligence extraction failed: {e}", exc_info=True)
            raise AzureServiceError(f"Failed to extract text from document: {str(e)}")


def ingest_document(file_content: bytes, config) -> str:
    """
    Ingest a document by extracting text using Azure Document Intelligence.
    
    This is a convenience wrapper around extract_text_from_document() that provides
    a higher-level interface for document ingestion in the RAG pipeline.
    
    Args:
        file_content: Raw file content (bytes)
        config: Application configuration with Azure Document Intelligence credentials
        
    Returns:
        Document text content (same as extract_text_from_document)
        
    Raises:
        AzureServiceError: If ingestion fails (propagated from extract_text_from_document)
    """
    logger.info("Ingesting document")
    return extract_text_from_document(file_content, config)

