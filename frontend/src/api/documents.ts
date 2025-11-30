/**
 * API client for document operations
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export interface Document {
  document_id: string;
  filename: string;
  file_size: number;
  mime_type?: string;
  upload_timestamp?: string;
  status: string;
  chunks_created?: number;
  storage_path: string;
  preview_image_path?: string;
  metadata?: Record<string, any>;
}

export interface DocumentListResponse {
  documents: Document[];
  total: number;
  limit: number;
  offset: number;
}

export interface DocumentFilters {
  search?: string;
  file_type?: string;
  status?: string;
  date_from?: string;
  date_to?: string;
  size_min?: number;
  size_max?: number;
  sort_by?: string;
  sort_order?: 'asc' | 'desc';
  limit?: number;
  offset?: number;
}

export async function uploadDocument(file: File): Promise<Document> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE_URL}/api/upload`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || `Failed to upload document: ${response.statusText}`);
  }

  const data = await response.json();
  // The upload endpoint returns UploadResponse, but we need to fetch the full document
  // For now, return a partial document. In a real implementation, you might want to
  // refetch the document or have the upload endpoint return the full document.
  return {
    document_id: data.document_id,
    filename: file.name,
    file_size: file.size,
    mime_type: file.type,
    status: data.status,
    chunks_created: data.chunks_created,
    storage_path: '',
  };
}

export async function listDocuments(filters: DocumentFilters = {}): Promise<DocumentListResponse> {
  const params = new URLSearchParams();
  
  if (filters.search) params.append('search', filters.search);
  if (filters.file_type) params.append('file_type', filters.file_type);
  if (filters.status) params.append('status', filters.status);
  if (filters.date_from) params.append('date_from', filters.date_from);
  if (filters.date_to) params.append('date_to', filters.date_to);
  if (filters.size_min !== undefined) params.append('size_min', filters.size_min.toString());
  if (filters.size_max !== undefined) params.append('size_max', filters.size_max.toString());
  if (filters.sort_by) params.append('sort_by', filters.sort_by);
  if (filters.sort_order) params.append('sort_order', filters.sort_order);
  params.append('limit', (filters.limit || 100).toString());
  params.append('offset', (filters.offset || 0).toString());

  const response = await fetch(`${API_BASE_URL}/api/documents?${params.toString()}`);

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || `Failed to list documents: ${response.statusText}`);
  }

  return response.json();
}

export async function getDocument(documentId: string): Promise<Document> {
  const response = await fetch(`${API_BASE_URL}/api/documents/${documentId}`);

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || `Failed to get document: ${response.statusText}`);
  }

  return response.json();
}

export async function downloadDocument(documentId: string): Promise<Blob> {
  const response = await fetch(`${API_BASE_URL}/api/documents/${documentId}/download`);

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || `Failed to download document: ${response.statusText}`);
  }

  return response.blob();
}

export function getPreviewUrl(documentId: string): string {
  return `${API_BASE_URL}/api/documents/${documentId}/preview`;
}

export async function deleteDocument(documentId: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/api/documents/${documentId}`, {
    method: 'DELETE',
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || `Failed to delete document: ${response.statusText}`);
  }
}

