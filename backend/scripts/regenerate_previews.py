#!/usr/bin/env python3
"""
Script to regenerate preview images for documents that don't have them.

This script:
1. Finds all documents without preview_image_path
2. Downloads the original file from storage
3. Generates a preview image
4. Uploads the preview to storage
5. Updates the document record with the preview path
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_eval.core.config import Config
from rag_eval.core.logging import get_logger
from rag_eval.db.connection import DatabaseConnection
from rag_eval.db.documents import DocumentService
from rag_eval.services.rag.supabase_storage import (
    download_document_from_storage,
    generate_image_preview
)

logger = get_logger("scripts.regenerate_previews")


def regenerate_previews():
    """Regenerate preview images for documents without them"""
    config = Config.from_env()
    
    # Connect to database
    db_conn = DatabaseConnection(config)
    db_conn.connect()
    doc_service = DocumentService(db_conn)
    
    # Get all documents
    all_docs = doc_service.list_documents(limit=1000)
    
    # Filter documents without preview
    docs_without_preview = [doc for doc in all_docs if not doc.preview_image_path]
    
    if not docs_without_preview:
        print("✅ All documents already have preview images!")
        return
    
    print(f"Found {len(docs_without_preview)} document(s) without preview images")
    print()
    
    success_count = 0
    error_count = 0
    
    for doc in docs_without_preview:
        print(f"Processing: {doc.filename} (ID: {doc.document_id})")
        
        try:
            # Download original file
            file_content = download_document_from_storage(doc.storage_path, config)
            
            # Generate preview
            preview_path = generate_image_preview(
                file_content,
                doc.document_id,
                doc.mime_type or "application/octet-stream",
                config
            )
            
            if preview_path:
                # Update document record
                doc_service.update_preview_path(doc.document_id, preview_path)
                print(f"  ✅ Generated preview: {preview_path}")
                success_count += 1
            else:
                print(f"  ⚠️  Could not generate preview (unsupported type or missing dependencies)")
                error_count += 1
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
            error_count += 1
        
        print()
    
    print("=" * 60)
    print(f"Summary:")
    print(f"  ✅ Success: {success_count}")
    print(f"  ❌ Errors: {error_count}")
    print(f"  📊 Total: {len(docs_without_preview)}")


if __name__ == "__main__":
    try:
        regenerate_previews()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to regenerate previews: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        sys.exit(1)

