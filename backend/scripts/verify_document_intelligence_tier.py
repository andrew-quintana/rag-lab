#!/usr/bin/env python3
"""Script to help verify Azure Document Intelligence pricing tier and diagnose extraction issues"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.config import Config
from src.services.rag.ingestion import extract_text_from_document

def verify_tier_and_extraction(pdf_path: str):
    """Verify tier and test extraction"""
    config = Config.from_env()
    
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        print(f"❌ File not found: {pdf_path}")
        return
    
    print(f"\n=== Azure Document Intelligence Tier Verification ===\n")
    print(f"📄 File: {pdf_file.name}")
    print(f"   Size: {pdf_file.stat().st_size:,} bytes ({pdf_file.stat().st_size / 1024 / 1024:.2f} MB)\n")
    
    # Read file
    with open(pdf_file, "rb") as f:
        file_content = f.read()
    
    print(f"🔍 Testing extraction...")
    try:
        extracted_text = extract_text_from_document(file_content, config)
        
        text_length = len(extracted_text)
        file_size_mb = pdf_file.stat().st_size / 1024 / 1024
        
        print(f"\n📊 Extraction Results:")
        print(f"   Extracted text length: {text_length:,} characters")
        print(f"   File size: {file_size_mb:.2f} MB")
        print(f"   Ratio: {text_length / (file_size_mb * 1024 * 1024) * 100:.4f}% of file size")
        
        # Heuristic: For a typical PDF, we expect at least 0.1% of file size to be text
        # For scanned PDFs, this might be lower, but still should be > 0.01%
        expected_min_text = file_size_mb * 1024 * 1024 * 0.0001  # 0.01% of file size
        
        print(f"\n🔍 Tier Analysis:")
        if text_length < 1000:
            print(f"   ⚠️  WARNING: Very short extraction ({text_length} chars)")
            print(f"   This suggests only first page(s) were processed")
        elif text_length < expected_min_text:
            print(f"   ⚠️  WARNING: Extraction seems low for file size")
            print(f"   Expected at least {expected_min_text:.0f} chars, got {text_length:,}")
        
        # Check if this matches Free tier behavior (first 2 pages only)
        if text_length < 2000 and file_size_mb > 0.5:
            print(f"\n   🎯 LIKELY FREE TIER (F0) PER-DOCUMENT LIMITATION:")
            print(f"   - Free tier has 500 pages/month quota BUT only processes first 2 pages per document")
            print(f"   - Your file is {file_size_mb:.2f} MB (likely multi-page, ~250 pages)")
            print(f"   - Only {text_length} chars extracted (consistent with 1-2 pages)")
            print(f"   - Even though 250 pages is within the 500-page monthly quota,")
            print(f"     Free tier will still only process the first 2 pages of this document")
            print(f"\n   ✅ RECOMMENDATION: Upgrade to Standard (S0) tier")
            print(f"   - Standard tier processes up to 2,000 pages per document")
            print(f"   - Check Azure Portal: Resource → Pricing tier")
        else:
            print(f"   ✅ Extraction length seems reasonable for file size")
        
        print(f"\n📋 How to Verify Tier in Azure Portal:")
        print(f"   1. Go to Azure Portal (portal.azure.com)")
        print(f"   2. Navigate to your Document Intelligence resource")
        print(f"   3. Go to 'Pricing tier' section")
        print(f"   4. Verify it's set to 'Standard (S0)' or higher")
        print(f"   5. Free (F0) tier has 2-page limit per document")
        
        print(f"\n📋 Tier Limits:")
        print(f"   Free (F0):")
        print(f"      - Monthly quota: 500 pages free per month")
        print(f"      - Per-document limit: First 2 pages only (key limitation)")
        print(f"      - Max document size: 4 MB")
        print(f"   Standard (S0):")
        print(f"      - Pay per page pricing")
        print(f"      - Per-document limit: Up to 2,000 pages per document")
        print(f"      - Max document size: 500 MB")
        print(f"   Premium: Custom limits")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        pdf_path = "docs/inputs/scan_classic_hmo.pdf"
        if not Path(pdf_path).exists():
            print("Usage: python verify_document_intelligence_tier.py <path_to_pdf>")
            sys.exit(1)
    else:
        pdf_path = sys.argv[1]
    
    verify_tier_and_extraction(pdf_path)

