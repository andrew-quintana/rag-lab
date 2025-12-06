#!/usr/bin/env python3
"""Test the batch extraction implementation in ingestion.py"""

import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_eval.core.config import Config
from rag_eval.services.rag.ingestion import extract_text_from_document

def main():
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "docs/inputs/scan_classic_hmo.pdf"
    
    if not Path(pdf_path).exists():
        print(f"❌ File not found: {pdf_path}")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print(f"Testing Batch Extraction Implementation")
    print(f"Document: {pdf_path}")
    print(f"{'='*70}\n")
    
    config = Config.from_env()
    
    with open(pdf_path, "rb") as f:
        file_content = f.read()
    
    print(f"📄 File size: {len(file_content):,} bytes\n")
    print("🔍 Extracting text (this may take several minutes for large documents)...\n")
    
    try:
        extracted_text = extract_text_from_document(file_content, config)
        
        print(f"\n{'='*70}")
        print("RESULTS")
        print(f"{'='*70}\n")
        print(f"✅ Extraction successful!")
        print(f"   Total characters: {len(extracted_text):,}")
        print(f"   Preview (first 500 chars):")
        print(f"   {extracted_text[:500]}...\n")
        
        # Compare with expected
        expected_min = 50_000  # Based on pdftotext showing 514K chars
        if len(extracted_text) > expected_min:
            print(f"✅ SUCCESS: Extracted {len(extracted_text):,} characters (expected > {expected_min:,})")
        else:
            print(f"⚠️  WARNING: Only extracted {len(extracted_text):,} characters (expected > {expected_min:,})")
            print(f"   This may indicate batch processing didn't work as expected.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

