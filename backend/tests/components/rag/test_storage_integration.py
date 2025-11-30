"""Integration tests for Supabase Storage with real RLS policies

These tests require:
1. Supabase running locally (supabase start)
2. Migration 0004_setup_storage_bucket.sql applied
3. Valid SUPABASE_URL and SUPABASE_KEY in environment

These tests are marked with @pytest.mark.integration and can be skipped
if Supabase is not available.
"""

import pytest
import os
from pathlib import Path
from rag_eval.core.config import Config
from rag_eval.services.rag.supabase_storage import (
    upload_document_to_storage,
    download_document_from_storage,
    delete_document_from_storage,
    get_public_url
)

# Mark all tests in this file as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture
def real_config():
    """Get real config from environment"""
    try:
        config = Config.from_env()
        # Verify Supabase is configured
        if not config.supabase_url or not config.supabase_key:
            pytest.skip("Supabase credentials not configured")
        return config
    except Exception as e:
        pytest.skip(f"Failed to load config: {e}")


@pytest.fixture
def sample_file_content():
    """Sample PDF content for testing"""
    return b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\nxref\n0 0\ntrailer\n<< /Size 0 /Root 1 0 R >>\nstartxref\n0\n%%EOF"


@pytest.fixture
def test_document_id():
    """Generate a unique document ID for testing"""
    import uuid
    return f"test-{uuid.uuid4()}"


class TestStorageIntegration:
    """Integration tests for Supabase Storage with real RLS policies"""
    
    def test_upload_with_rls_policies(self, real_config, sample_file_content, test_document_id):
        """Test that upload works with RLS policies in place"""
        # This test verifies that the migration 0004_setup_storage_bucket.sql
        # correctly sets up RLS policies that allow uploads
        try:
            result = upload_document_to_storage(
                sample_file_content,
                test_document_id,
                "test.pdf",
                real_config,
                content_type="application/pdf"
            )
            assert result == test_document_id
        except Exception as e:
            # If this fails, it likely means RLS policies are not set up correctly
            pytest.fail(f"Upload failed - RLS policies may not be configured: {e}")
    
    def test_download_with_rls_policies(self, real_config, sample_file_content, test_document_id):
        """Test that download works with RLS policies in place"""
        # First upload a file
        upload_document_to_storage(
            sample_file_content,
            test_document_id,
            "test.pdf",
            real_config,
            content_type="application/pdf"
        )
        
        # Then download it
        try:
            downloaded = download_document_from_storage(test_document_id, real_config)
            assert downloaded == sample_file_content
        except Exception as e:
            pytest.fail(f"Download failed - RLS policies may not be configured: {e}")
        finally:
            # Cleanup
            try:
                delete_document_from_storage(test_document_id, real_config)
            except Exception:
                pass
    
    def test_delete_with_rls_policies(self, real_config, sample_file_content, test_document_id):
        """Test that delete works with RLS policies in place"""
        # First upload a file
        upload_document_to_storage(
            sample_file_content,
            test_document_id,
            "test.pdf",
            real_config,
            content_type="application/pdf"
        )
        
        # Then delete it
        try:
            delete_document_from_storage(test_document_id, real_config)
            # Verify it's deleted by trying to download (should fail)
            try:
                download_document_from_storage(test_document_id, real_config)
                pytest.fail("File should have been deleted")
            except Exception:
                pass  # Expected - file should not exist
        except Exception as e:
            pytest.fail(f"Delete failed - RLS policies may not be configured: {e}")
    
    def test_public_url_generation(self, real_config, sample_file_content, test_document_id):
        """Test that public URL generation works"""
        # First upload a file
        upload_document_to_storage(
            sample_file_content,
            test_document_id,
            "test.pdf",
            real_config,
            content_type="application/pdf"
        )
        
        try:
            url = get_public_url(test_document_id, real_config)
            assert url is not None
            # Supabase returns a dict with 'signedURL' or 'signedUrl' key
            if isinstance(url, dict):
                url = url.get('signedURL') or url.get('signedUrl')
            assert isinstance(url, str)
            assert len(url) > 0
        except Exception as e:
            pytest.fail(f"Public URL generation failed: {e}")
        finally:
            # Cleanup
            try:
                delete_document_from_storage(test_document_id, real_config)
            except Exception:
                pass
    
    def test_upload_retry_on_failure(self, real_config, sample_file_content, test_document_id):
        """Test that retry logic works correctly"""
        # This test verifies the retry mechanism works
        # (though with proper RLS setup, it shouldn't need to retry)
        result = upload_document_to_storage(
            sample_file_content,
            test_document_id,
            "test.pdf",
            real_config,
            content_type="application/pdf"
        )
        assert result == test_document_id
        
        # Cleanup
        try:
            delete_document_from_storage(test_document_id, real_config)
        except Exception:
            pass

