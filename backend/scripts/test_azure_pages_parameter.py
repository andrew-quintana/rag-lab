#!/usr/bin/env python3
"""Test script to investigate Azure Document Intelligence page extraction limits

This script tests various API parameters and models to find a way to extract
more than 2 pages from a 232-page PDF document.
"""

import sys
import os
from pathlib import Path
from io import BytesIO

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_eval.core.config import Config
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest

def test_extraction(pdf_path: str, model_id: str, pages: str = None, use_request_object: bool = True, description: str = ""):
    """Test extraction with specific parameters"""
    config = Config.from_env()
    
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        print(f"❌ File not found: {pdf_path}")
        return None
    
    print(f"\n{'='*70}")
    print(f"Test: {description}")
    print(f"Model: {model_id}")
    print(f"Pages parameter: {pages}")
    print(f"Use AnalyzeDocumentRequest: {use_request_object}")
    print(f"{'='*70}\n")
    
    # Read file
    with open(pdf_file, "rb") as f:
        file_content = f.read()
    
    try:
        # Initialize client
        client = DocumentIntelligenceClient(
            endpoint=config.azure_document_intelligence_endpoint,
            credential=AzureKeyCredential(config.azure_document_intelligence_api_key)
        )
        
        # Prepare request
        if use_request_object:
            try:
                request = AnalyzeDocumentRequest(bytes_source=file_content)
                poller = client.begin_analyze_document(
                    model_id=model_id,
                    analyze_request=request,
                    pages=pages
                )
            except (TypeError, AttributeError) as e:
                print(f"⚠️  AnalyzeDocumentRequest failed: {e}")
                print("   Falling back to direct body parameter...")
                poller = client.begin_analyze_document(
                    model_id=model_id,
                    body=BytesIO(file_content),
                    pages=pages
                )
        else:
            poller = client.begin_analyze_document(
                model_id=model_id,
                body=BytesIO(file_content),
                pages=pages
            )
        
        print("⏳ Polling for results (timeout: 600s)...")
        result = poller.result(timeout=600)
        
        # Extract text
        page_texts = []
        if hasattr(result, 'pages') and result.pages:
            print(f"📄 Pages detected: {len(result.pages)}")
            for page_num, page in enumerate(result.pages, 1):
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
                    page_text = "\n".join(page_lines)
                    page_texts.append(page_text)
                    print(f"   Page {page_num}: {len(page_text)} characters")
                else:
                    print(f"   Page {page_num}: No text extracted")
        
        # Also check content field
        content_text = ""
        if hasattr(result, 'content') and result.content:
            content_text = result.content
            print(f"📝 Content field: {len(content_text)} characters")
        
        # Combine all text
        all_text_parts = []
        if page_texts:
            all_text_parts.append("\n\n".join(page_texts))
        if content_text:
            all_text_parts.append(content_text)
        
        full_text = "\n\n".join(all_text_parts)
        total_chars = len(full_text)
        
        print(f"\n✅ Result:")
        print(f"   Pages with text: {len(page_texts)}")
        print(f"   Total characters extracted: {total_chars:,}")
        print(f"   Preview (first 200 chars): {full_text[:200]}...")
        
        return {
            'pages_detected': len(result.pages) if hasattr(result, 'pages') and result.pages else 0,
            'pages_with_text': len(page_texts),
            'total_chars': total_chars,
            'text': full_text
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "docs/inputs/scan_classic_hmo.pdf"
    
    if not Path(pdf_path).exists():
        print(f"❌ File not found: {pdf_path}")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print(f"Azure Document Intelligence Page Extraction Test")
    print(f"Document: {pdf_path}")
    print(f"{'='*70}\n")
    
    results = []
    
    # Test 1: Default (pages=None) with prebuilt-read
    result = test_extraction(
        pdf_path,
        model_id="prebuilt-read",
        pages=None,
        use_request_object=True,
        description="Test 1: Default (pages=None) with prebuilt-read"
    )
    if result:
        results.append(("Test 1: Default", result))
    
    # Test 2: Explicit page range 1-10
    result = test_extraction(
        pdf_path,
        model_id="prebuilt-read",
        pages="1-10",
        use_request_object=True,
        description="Test 2: Explicit page range 1-10"
    )
    if result:
        results.append(("Test 2: Pages 1-10", result))
    
    # Test 3: Explicit page range 3-5 (to test if we can skip first 2 pages)
    result = test_extraction(
        pdf_path,
        model_id="prebuilt-read",
        pages="3-5",
        use_request_object=True,
        description="Test 3: Explicit page range 3-5"
    )
    if result:
        results.append(("Test 3: Pages 3-5", result))
    
    # Test 4: Try prebuilt-layout model
    result = test_extraction(
        pdf_path,
        model_id="prebuilt-layout",
        pages=None,
        use_request_object=True,
        description="Test 4: prebuilt-layout model (pages=None)"
    )
    if result:
        results.append(("Test 4: prebuilt-layout", result))
    
    # Test 5: Try prebuilt-layout with explicit page range
    result = test_extraction(
        pdf_path,
        model_id="prebuilt-layout",
        pages="1-10",
        use_request_object=True,
        description="Test 5: prebuilt-layout model (pages=1-10)"
    )
    if result:
        results.append(("Test 5: prebuilt-layout 1-10", result))
    
    # Test 6: Try prebuilt-document model
    result = test_extraction(
        pdf_path,
        model_id="prebuilt-document",
        pages=None,
        use_request_object=True,
        description="Test 6: prebuilt-document model (pages=None)"
    )
    if result:
        results.append(("Test 6: prebuilt-document", result))
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")
    print(f"{'Test':<30} {'Pages':<10} {'Chars':<15}")
    print(f"{'-'*70}")
    for test_name, result in results:
        print(f"{test_name:<30} {result['pages_with_text']:<10} {result['total_chars']:,}")

if __name__ == "__main__":
    main()

