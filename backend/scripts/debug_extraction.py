#!/usr/bin/env python3
"""Debug script to inspect Azure Document Intelligence result structure"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_eval.core.config import Config
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from io import BytesIO

def debug_extraction(pdf_path: str):
    """Debug extraction to see what's in the result"""
    config = Config.from_env()
    
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        print(f"❌ File not found: {pdf_path}")
        return
    
    print(f"\n=== Debugging Azure Document Intelligence Extraction ===\n")
    print(f"📄 File: {pdf_file.name}")
    print(f"   Size: {pdf_file.stat().st_size:,} bytes\n")
    
    # Read file
    with open(pdf_file, "rb") as f:
        file_content = f.read()
    
    try:
        # Initialize client
        client = DocumentIntelligenceClient(
            endpoint=config.azure_document_intelligence_endpoint,
            credential=AzureKeyCredential(config.azure_document_intelligence_api_key)
        )
        
        # Analyze document
        print("🔍 Analyzing document with Azure Document Intelligence...")
        poller = client.begin_analyze_document(
            model_id="prebuilt-read",
            body=BytesIO(file_content)
        )
        
        result = poller.result()
        
        print(f"\n📊 Result Structure:\n")
        print(f"Result type: {type(result)}")
        print(f"Result attributes: {dir(result)}\n")
        
        # Check content
        print(f"✅ result.content:")
        print(f"   Type: {type(result.content)}")
        print(f"   Length: {len(result.content) if result.content else 0}")
        print(f"   Preview: {result.content[:200] if result.content else 'None'}\n")
        
        # Check pages
        if hasattr(result, 'pages'):
            print(f"✅ result.pages:")
            print(f"   Type: {type(result.pages)}")
            print(f"   Count: {len(result.pages) if result.pages else 0}")
            if result.pages:
                for i, page in enumerate(result.pages[:3], 1):
                    print(f"\n   Page {i}:")
                    print(f"      Type: {type(page)}")
                    print(f"      Attributes: {[a for a in dir(page) if not a.startswith('_')]}")
                    
                    if hasattr(page, 'lines'):
                        print(f"      Lines: {len(page.lines) if page.lines else 0}")
                        if page.lines:
                            print(f"      First 3 lines:")
                            for line in page.lines[:3]:
                                if hasattr(line, 'content'):
                                    print(f"         - {line.content[:100]}")
                    
                    if hasattr(page, 'paragraphs'):
                        print(f"      Paragraphs: {len(page.paragraphs) if page.paragraphs else 0}")
                        if page.paragraphs:
                            print(f"      First paragraph: {page.paragraphs[0].content[:200] if hasattr(page.paragraphs[0], 'content') else 'N/A'}")
                    
                    if hasattr(page, 'words'):
                        print(f"      Words: {len(page.words) if page.words else 0}")
        else:
            print(f"❌ result.pages: Not available\n")
        
        # Check paragraphs (top-level)
        if hasattr(result, 'paragraphs'):
            print(f"✅ result.paragraphs:")
            print(f"   Type: {type(result.paragraphs)}")
            print(f"   Count: {len(result.paragraphs) if result.paragraphs else 0}")
            if result.paragraphs:
                for i, para in enumerate(result.paragraphs[:3], 1):
                    print(f"   Paragraph {i}: {para.content[:200] if hasattr(para, 'content') else 'N/A'}")
        else:
            print(f"❌ result.paragraphs: Not available\n")
        
        # Check tables
        if hasattr(result, 'tables'):
            print(f"✅ result.tables:")
            print(f"   Type: {type(result.tables)}")
            print(f"   Count: {len(result.tables) if result.tables else 0}\n")
        else:
            print(f"❌ result.tables: Not available\n")
        
        # Try to extract all text from pages
        print(f"\n🔍 Attempting to extract all text from pages...")
        all_text_parts = []
        
        if hasattr(result, 'pages') and result.pages:
            for page_num, page in enumerate(result.pages, 1):
                page_text = []
                
                # Try lines
                if hasattr(page, 'lines') and page.lines:
                    for line in page.lines:
                        if hasattr(line, 'content') and line.content:
                            page_text.append(line.content)
                
                # Try paragraphs
                if hasattr(page, 'paragraphs') and page.paragraphs:
                    for para in page.paragraphs:
                        if hasattr(para, 'content') and para.content:
                            page_text.append(para.content)
                
                # Try words (fallback)
                if not page_text and hasattr(page, 'words') and page.words:
                    words = [w.content for w in page.words if hasattr(w, 'content') and w.content]
                    if words:
                        page_text.append(" ".join(words))
                
                if page_text:
                    all_text_parts.append(f"=== Page {page_num} ===\n" + "\n".join(page_text))
        
        if all_text_parts:
            full_text = "\n\n".join(all_text_parts)
            print(f"   ✅ Extracted {len(full_text):,} characters from page structure")
            print(f"   Preview (first 500 chars):\n{full_text[:500]}\n")
        else:
            print(f"   ❌ Could not extract text from page structure")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        pdf_path = "docs/inputs/scan_classic_hmo.pdf"
        if not Path(pdf_path).exists():
            print("Usage: python debug_extraction.py <path_to_pdf>")
            sys.exit(1)
    else:
        pdf_path = sys.argv[1]
    
    debug_extraction(pdf_path)

