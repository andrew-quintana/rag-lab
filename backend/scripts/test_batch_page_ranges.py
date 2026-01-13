#!/usr/bin/env python3
"""Test batch processing with different page ranges to extract all pages"""

import sys
import os
from pathlib import Path
from io import BytesIO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.config import Config
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential

def extract_page_range(pdf_path: str, page_range: str, model_id: str = "prebuilt-read"):
    """Extract text from a specific page range"""
    config = Config.from_env()
    
    with open(pdf_path, "rb") as f:
        file_content = f.read()
    
    try:
        client = DocumentIntelligenceClient(
            endpoint=config.azure_document_intelligence_endpoint,
            credential=AzureKeyCredential(config.azure_document_intelligence_api_key)
        )
        
        poller = client.begin_analyze_document(
            model_id=model_id,
            body=BytesIO(file_content),
            pages=page_range
        )
        
        result = poller.result(timeout=600)
        
        # Extract text from pages
        page_texts = []
        if hasattr(result, 'pages') and result.pages:
            for page in result.pages:
                page_lines = []
                
                if hasattr(page, 'lines') and page.lines:
                    for line in page.lines:
                        if hasattr(line, 'content') and line.content:
                            page_lines.append(line.content)
                
                if not page_lines and hasattr(page, 'paragraphs') and page.paragraphs:
                    for para in page.paragraphs:
                        if hasattr(para, 'content') and para.content:
                            page_lines.append(para.content)
                
                if not page_lines and hasattr(page, 'words') and page.words:
                    words = [w.content for w in page.words if hasattr(w, 'content') and w.content]
                    if words:
                        page_lines.append(" ".join(words))
                
                if page_lines:
                    page_texts.append("\n".join(page_lines))
        
        # Also get content field
        content_text = ""
        if hasattr(result, 'content') and result.content:
            content_text = result.content
        
        # Combine
        all_text = []
        if page_texts:
            all_text.append("\n\n".join(page_texts))
        if content_text:
            all_text.append(content_text)
        
        full_text = "\n\n".join(all_text)
        
        return {
            'pages_detected': len(result.pages) if hasattr(result, 'pages') and result.pages else 0,
            'chars': len(full_text),
            'text': full_text
        }
        
    except Exception as e:
        print(f"❌ Error extracting {page_range}: {e}")
        return None

def main():
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "docs/inputs/scan_classic_hmo.pdf"
    
    print(f"\n{'='*70}")
    print(f"Batch Page Range Extraction Test")
    print(f"Document: {pdf_path}")
    print(f"{'='*70}\n")
    
    # Test various page ranges
    test_ranges = [
        "1-2",    # First 2 pages (should work, but limited)
        "3-4",    # Pages 3-4 (should work based on Test 3)
        "5-6",    # Pages 5-6
        "10-11",  # Pages 10-11
        "50-51",  # Pages 50-51
        "100-101", # Pages 100-101
        "200-201", # Pages 200-201
        "230-232", # Last pages
    ]
    
    results = []
    for page_range in test_ranges:
        print(f"Testing pages {page_range}...", end=" ")
        result = extract_page_range(pdf_path, page_range)
        if result:
            print(f"✅ {result['pages_detected']} pages, {result['chars']:,} chars")
            results.append((page_range, result))
        else:
            print("❌ Failed")
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")
    print(f"{'Page Range':<15} {'Pages':<10} {'Chars':<15}")
    print(f"{'-'*70}")
    for page_range, result in results:
        print(f"{page_range:<15} {result['pages_detected']:<10} {result['chars']:,}")

if __name__ == "__main__":
    main()







