#!/usr/bin/env python3
"""Script to test document extraction and chunking to diagnose why only one chunk is created"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.config import Config
from src.services.rag.ingestion import ingest_document
from src.services.rag.chunking import chunk_text

def test_extraction_and_chunking(pdf_path: str):
    """Test extraction and chunking for a PDF file"""
    config = Config.from_env()
    
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        print(f"❌ File not found: {pdf_path}")
        return
    
    print(f"\n=== Testing Extraction and Chunking ===\n")
    print(f"📄 File: {pdf_file.name}")
    print(f"   Size: {pdf_file.stat().st_size:,} bytes ({pdf_file.stat().st_size / 1024 / 1024:.2f} MB)")
    
    # Read file
    with open(pdf_file, "rb") as f:
        file_content = f.read()
    
    print(f"\n🔍 Step 1: Extracting text using Azure Document Intelligence...")
    try:
        extracted_text = ingest_document(file_content, config)
        
        text_length = len(extracted_text)
        text_length_no_whitespace = len(extracted_text.replace(" ", "").replace("\n", "").replace("\t", ""))
        
        print(f"   ✅ Extraction successful")
        print(f"   📊 Extracted text length: {text_length:,} characters")
        print(f"   📊 Text length (no whitespace): {text_length_no_whitespace:,} characters")
        
        # Show preview
        preview_length = min(500, text_length)
        print(f"\n   📝 Text preview (first {preview_length} chars):")
        print(f"   {'-' * 60}")
        print(f"   {extracted_text[:preview_length]}")
        if text_length > preview_length:
            print(f"   ... ({text_length - preview_length:,} more characters)")
        print(f"   {'-' * 60}")
        
        # Check if text is suspiciously short
        if text_length < 1000:
            print(f"\n   ⚠️  WARNING: Extracted text is very short ({text_length} chars)")
            print(f"   This would result in only 1 chunk (chunk_size default is 1000)")
            print(f"   Possible causes:")
            print(f"   - PDF is mostly images/scans with minimal text")
            print(f"   - OCR extraction failed or extracted very little")
            print(f"   - Document structure issue")
        elif text_length < 10000:
            print(f"\n   ⚠️  WARNING: Extracted text is relatively short ({text_length} chars)")
            print(f"   For a {pdf_file.stat().st_size / 1024 / 1024:.2f} MB file, this seems low")
        
    except Exception as e:
        print(f"   ❌ Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\n🔍 Step 2: Chunking extracted text...")
    try:
        # Use default chunking parameters
        chunks = chunk_text(extracted_text, config, document_id="test-doc")
        
        print(f"   ✅ Chunking successful")
        print(f"   📊 Number of chunks created: {len(chunks)}")
        
        if len(chunks) == 1:
            print(f"\n   ⚠️  WARNING: Only 1 chunk created!")
            print(f"   This is because extracted text ({text_length} chars) is shorter than chunk_size (1000)")
            print(f"   Chunk details:")
            chunk = chunks[0]
            print(f"   - Chunk ID: {chunk.chunk_id}")
            print(f"   - Chunk length: {len(chunk.text):,} characters")
            print(f"   - Metadata: {chunk.metadata}")
        else:
            print(f"\n   ✅ Multiple chunks created as expected")
            print(f"   Chunk details:")
            for i, chunk in enumerate(chunks[:5], 1):
                print(f"   {i}. {chunk.chunk_id}: {len(chunk.text):,} chars")
            if len(chunks) > 5:
                print(f"   ... and {len(chunks) - 5} more chunks")
        
        # Calculate expected chunks
        chunk_size = 1000
        overlap = 200
        effective_chunk_size = chunk_size - overlap  # 800 chars per chunk on average
        expected_chunks = max(1, (text_length + overlap - 1) // effective_chunk_size)
        
        print(f"\n   📊 Expected chunks (for {text_length:,} chars):")
        print(f"   - Chunk size: {chunk_size} chars")
        print(f"   - Overlap: {overlap} chars")
        print(f"   - Effective size: {effective_chunk_size} chars per chunk")
        print(f"   - Expected: ~{expected_chunks} chunks")
        
        if len(chunks) != expected_chunks:
            print(f"   ⚠️  Mismatch: Got {len(chunks)} chunks, expected ~{expected_chunks}")
        else:
            print(f"   ✅ Chunk count matches expectation")
        
    except Exception as e:
        print(f"   ❌ Chunking failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Try to find the scan_classic_hmo.pdf file
        possible_paths = [
            "docs/inputs/scan_classic_hmo.pdf",
            "../docs/inputs/scan_classic_hmo.pdf",
            "../../docs/inputs/scan_classic_hmo.pdf",
        ]
        
        pdf_path = None
        for path in possible_paths:
            if Path(path).exists():
                pdf_path = path
                break
        
        if not pdf_path:
            print("Usage: python test_extraction_chunking.py <path_to_pdf>")
            print("\nOr place scan_classic_hmo.pdf in docs/inputs/")
            sys.exit(1)
    else:
        pdf_path = sys.argv[1]
    
    test_extraction_and_chunking(pdf_path)




