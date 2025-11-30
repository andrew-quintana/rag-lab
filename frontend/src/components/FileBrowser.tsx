import { useState, useEffect, useCallback } from 'react';
import { Download, Trash2, Eye, ChevronUp, ChevronDown, Image as ImageIcon, CheckSquare, Square, Search } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import {
  listDocuments,
  deleteDocument,
  downloadDocument,
  getPreviewUrl,
  getDocument,
  type Document,
  type DocumentFilters,
} from '@/api/documents';
import { formatDate, formatMimeType } from '@/lib/utils';

interface FileBrowserProps {
  refreshTrigger?: number;
}

type SortField = 'upload_timestamp' | 'filename' | 'file_size' | 'mime_type' | 'status';
type SortOrder = 'asc' | 'desc';

export function FileBrowser({ refreshTrigger }: FileBrowserProps) {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(0);
  const pageSize = 20;

  // Filters
  const [search, setSearch] = useState('');
  const [fileType, setFileType] = useState<string>('all');
  const [status, setStatus] = useState<string>('all');
  const [dateFrom, setDateFrom] = useState('');
  const [dateTo, setDateTo] = useState('');
  const [sizeMin, setSizeMin] = useState('');
  const [sizeMax, setSizeMax] = useState('');

  // Sorting
  const [sortBy, setSortBy] = useState<SortField>('upload_timestamp');
  const [sortOrder, setSortOrder] = useState<SortOrder>('desc');

  // Dialogs
  const [selectedDocument, setSelectedDocument] = useState<Document | null>(null);
  const [selectedDocuments, setSelectedDocuments] = useState<Set<string>>(new Set());
  const [showMetadataDialog, setShowMetadataDialog] = useState(false);
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const [showPreviewLightbox, setShowPreviewLightbox] = useState(false);
  const [previewLightboxUrl, setPreviewLightboxUrl] = useState<string | null>(null);
  const [imageErrors, setImageErrors] = useState<Set<string>>(new Set());
  const [isDeleting, setIsDeleting] = useState(false);
  const [filtersExpanded, setFiltersExpanded] = useState(false);

  const loadDocuments = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const filters: DocumentFilters = {
        search: search || undefined,
        file_type: fileType && fileType !== 'all' ? fileType : undefined,
        status: status && status !== 'all' ? status : undefined,
        date_from: dateFrom || undefined,
        date_to: dateTo || undefined,
        size_min: sizeMin ? parseInt(sizeMin) : undefined,
        size_max: sizeMax ? parseInt(sizeMax) : undefined,
        sort_by: sortBy,
        sort_order: sortOrder,
        limit: pageSize,
        offset: page * pageSize,
      };

      const response = await listDocuments(filters);
      setDocuments(response.documents);
      setTotal(response.total);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load documents');
    } finally {
      setLoading(false);
    }
  }, [search, fileType, status, dateFrom, dateTo, sizeMin, sizeMax, sortBy, sortOrder, page]);

  useEffect(() => {
    loadDocuments();
    // Clear selection when documents reload
    setSelectedDocuments(new Set());
  }, [loadDocuments, refreshTrigger]);

  const handleSort = (field: SortField) => {
    if (sortBy === field) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortBy(field);
      setSortOrder('asc');
    }
    setPage(0);
  };

  const handleDelete = async () => {
    const docsToDelete = selectedDocument 
      ? [selectedDocument] 
      : documents.filter(doc => selectedDocuments.has(doc.document_id));
    
    if (docsToDelete.length === 0) return;

    setIsDeleting(true);
    try {
      // Delete all selected documents
      const deletePromises = docsToDelete.map(doc => deleteDocument(doc.document_id));
      await Promise.all(deletePromises);
      
      setShowDeleteDialog(false);
      setSelectedDocument(null);
      setSelectedDocuments(new Set());
      loadDocuments();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete document(s)');
    } finally {
      setIsDeleting(false);
    }
  };

  const handleToggleSelect = (documentId: string) => {
    setSelectedDocuments(prev => {
      const next = new Set(prev);
      if (next.has(documentId)) {
        next.delete(documentId);
      } else {
        next.add(documentId);
      }
      return next;
    });
  };

  const handleSelectAll = () => {
    if (selectedDocuments.size === documents.length) {
      setSelectedDocuments(new Set());
    } else {
      setSelectedDocuments(new Set(documents.map(doc => doc.document_id)));
    }
  };

  const handleDownload = async (doc: Document) => {
    try {
      const blob = await downloadDocument(doc.document_id);
      const url = window.URL.createObjectURL(blob);
      const a = window.document.createElement('a');
      a.href = url;
      a.download = doc.filename;
      window.document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      window.document.body.removeChild(a);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to download document');
    }
  };

  const handleViewMetadata = async (doc: Document) => {
    try {
      const fullDoc = await getDocument(doc.document_id);
      setSelectedDocument(fullDoc);
      setShowMetadataDialog(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load document metadata');
    }
  };

  const handleOpenPreview = (doc: Document) => {
    if (doc.preview_image_path) {
      // Close metadata dialog if open
      setShowMetadataDialog(false);
      // Open preview lightbox
      setPreviewLightboxUrl(getPreviewUrl(doc.document_id));
      setShowPreviewLightbox(true);
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
  };

  const getStatusBadgeVariant = (status: string) => {
    switch (status) {
      case 'processed':
        return 'default';
      case 'processing':
        return 'secondary';
      case 'uploaded':
        return 'outline';
      default:
        return 'outline';
    }
  };

  const SortIcon = ({ field }: { field: SortField }) => {
    if (sortBy !== field) return null;
    return sortOrder === 'asc' ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />;
  };

  const totalPages = Math.ceil(total / pageSize);

  return (
    <div className="space-y-4">
      {/* Filters */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle>Filters</CardTitle>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setFiltersExpanded(!filtersExpanded)}
              className="h-8 w-8"
            >
              {filtersExpanded ? (
                <ChevronUp className="h-4 w-4" />
              ) : (
                <ChevronDown className="h-4 w-4" />
              )}
            </Button>
          </div>
        </CardHeader>
        {filtersExpanded && (
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="space-y-2">
              <Label htmlFor="file-type">File Type</Label>
              <Select value={fileType} onValueChange={(value) => {
                setFileType(value);
                setPage(0);
              }}>
                <SelectTrigger id="file-type">
                  <SelectValue placeholder="All types" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All types</SelectItem>
                  <SelectItem value="application/pdf">PDF</SelectItem>
                  <SelectItem value="image/jpeg">JPEG</SelectItem>
                  <SelectItem value="image/png">PNG</SelectItem>
                  <SelectItem value="image/gif">GIF</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="status">Status</Label>
              <Select value={status} onValueChange={(value) => {
                setStatus(value);
                setPage(0);
              }}>
                <SelectTrigger id="status">
                  <SelectValue placeholder="All statuses" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All statuses</SelectItem>
                  <SelectItem value="uploaded">Uploaded</SelectItem>
                  <SelectItem value="processing">Processing</SelectItem>
                  <SelectItem value="processed">Processed</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="date-from">Date From</Label>
              <Input
                id="date-from"
                type="date"
                value={dateFrom}
                onChange={(e) => {
                  setDateFrom(e.target.value);
                  setPage(0);
                }}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="date-to">Date To</Label>
              <Input
                id="date-to"
                type="date"
                value={dateTo}
                onChange={(e) => {
                  setDateTo(e.target.value);
                  setPage(0);
                }}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="size-min">Min Size (bytes)</Label>
              <Input
                id="size-min"
                type="number"
                placeholder="Min"
                value={sizeMin}
                onChange={(e) => {
                  setSizeMin(e.target.value);
                  setPage(0);
                }}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="size-max">Max Size (bytes)</Label>
              <Input
                id="size-max"
                type="number"
                placeholder="Max"
                value={sizeMax}
                onChange={(e) => {
                  setSizeMax(e.target.value);
                  setPage(0);
                }}
              />
            </div>
          </div>
        </CardContent>
        )}
      </Card>

      {/* Table */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between gap-4">
            <CardTitle className="flex-shrink-0">Documents ({total})</CardTitle>
            <div className="flex items-center gap-2 flex-1 max-w-sm">
              <Search className="h-4 w-4 text-muted-foreground flex-shrink-0" />
              <Input
                id="search"
                placeholder="Search filename..."
                value={search}
                onChange={(e) => {
                  setSearch(e.target.value);
                  setPage(0);
                }}
                className="flex-1"
              />
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {error && (
            <div className="mb-4 p-4 bg-destructive/10 text-destructive rounded-md">
              {error}
            </div>
          )}

          {loading ? (
            <div className="text-center py-8">Loading...</div>
          ) : documents.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground">No documents found</div>
          ) : (
            <>
              <div className="rounded-md border">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead className="w-12">
                        <button
                          onClick={handleSelectAll}
                          className="flex items-center justify-center"
                          title={selectedDocuments.size === documents.length && documents.length > 0 ? "Deselect all" : "Select all"}
                        >
                          {selectedDocuments.size === documents.length && documents.length > 0 ? (
                            <CheckSquare className="h-4 w-4" />
                          ) : (
                            <Square className="h-4 w-4" />
                          )}
                        </button>
                      </TableHead>
                      <TableHead className="w-16">Preview</TableHead>
                      <TableHead
                        className="cursor-pointer"
                        onClick={() => handleSort('filename')}
                      >
                        <div className="flex items-center gap-2">
                          Filename
                          <SortIcon field="filename" />
                        </div>
                      </TableHead>
                      <TableHead
                        className="cursor-pointer"
                        onClick={() => handleSort('mime_type')}
                      >
                        <div className="flex items-center gap-2">
                          Type
                          <SortIcon field="mime_type" />
                        </div>
                      </TableHead>
                      <TableHead
                        className="cursor-pointer"
                        onClick={() => handleSort('file_size')}
                      >
                        <div className="flex items-center gap-2">
                          Size
                          <SortIcon field="file_size" />
                        </div>
                      </TableHead>
                      <TableHead
                        className="cursor-pointer"
                        onClick={() => handleSort('upload_timestamp')}
                      >
                        <div className="flex items-center gap-2">
                          Upload Date
                          <SortIcon field="upload_timestamp" />
                        </div>
                      </TableHead>
                      <TableHead
                        className="cursor-pointer"
                        onClick={() => handleSort('status')}
                      >
                        <div className="flex items-center gap-2">
                          Status
                          <SortIcon field="status" />
                        </div>
                      </TableHead>
                      <TableHead>Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {documents.map((doc) => (
                      <TableRow key={doc.document_id}>
                        <TableCell>
                          <button
                            onClick={() => handleToggleSelect(doc.document_id)}
                            className="flex items-center justify-center"
                            title={selectedDocuments.has(doc.document_id) ? "Deselect" : "Select"}
                          >
                            {selectedDocuments.has(doc.document_id) ? (
                              <CheckSquare className="h-4 w-4" />
                            ) : (
                              <Square className="h-4 w-4" />
                            )}
                          </button>
                        </TableCell>
                        <TableCell>
                          {doc.preview_image_path && !imageErrors.has(doc.document_id) ? (
                            <button
                              onClick={() => handleOpenPreview(doc)}
                              className="w-16 h-16 rounded border border-border hover:border-primary transition-colors overflow-hidden cursor-pointer bg-muted flex items-center justify-center group"
                              title="Click to view larger preview"
                            >
                              <img
                                src={getPreviewUrl(doc.document_id)}
                                alt={`Preview of ${doc.filename}`}
                                className="w-full h-full object-cover group-hover:opacity-90"
                                onError={() => {
                                  setImageErrors(prev => new Set(prev).add(doc.document_id));
                                }}
                              />
                            </button>
                          ) : (
                            <div className="w-16 h-16 bg-muted rounded flex items-center justify-center border border-border">
                              <ImageIcon className="h-4 w-4 text-muted-foreground" />
                            </div>
                          )}
                        </TableCell>
                        <TableCell className="font-medium">{doc.filename}</TableCell>
                        <TableCell>{formatMimeType(doc.mime_type)}</TableCell>
                        <TableCell>{formatFileSize(doc.file_size)}</TableCell>
                        <TableCell>
                          {doc.upload_timestamp ? formatDate(doc.upload_timestamp) : 'N/A'}
                        </TableCell>
                        <TableCell>
                          <Badge variant={getStatusBadgeVariant(doc.status) as any}>
                            {doc.status}
                          </Badge>
                        </TableCell>
                        <TableCell>
                          <div className="flex items-center gap-2">
                            <Button
                              variant="ghost"
                              size="icon"
                              onClick={() => handleViewMetadata(doc)}
                              title="View metadata"
                            >
                              <Eye className="h-4 w-4" />
                            </Button>
                            <Button
                              variant="ghost"
                              size="icon"
                              onClick={() => handleDownload(doc)}
                              title="Download"
                            >
                              <Download className="h-4 w-4" />
                            </Button>
                            <Button
                              variant="ghost"
                              size="icon"
                              onClick={() => {
                                setSelectedDocument(doc);
                                setSelectedDocuments(new Set([doc.document_id]));
                                setShowDeleteDialog(true);
                              }}
                              title="Delete"
                            >
                              <Trash2 className="h-4 w-4" />
                            </Button>
                          </div>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>

              {/* Pagination */}
              <div className="flex items-center justify-between mt-4">
                <div className="text-sm text-muted-foreground">
                  Showing {page * pageSize + 1} to {Math.min((page + 1) * pageSize, total)} of {total}
                </div>
                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    onClick={() => setPage(p => Math.max(0, p - 1))}
                    disabled={page === 0}
                  >
                    Previous
                  </Button>
                  <Button
                    variant="outline"
                    onClick={() => setPage(p => Math.min(totalPages - 1, p + 1))}
                    disabled={page >= totalPages - 1}
                  >
                    Next
                  </Button>
                </div>
              </div>
            </>
          )}
        </CardContent>
      </Card>

      {/* Metadata Dialog */}
      <Dialog open={showMetadataDialog} onOpenChange={setShowMetadataDialog}>
        <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Document Metadata</DialogTitle>
            <DialogDescription>
              Detailed information about {selectedDocument?.filename}
            </DialogDescription>
          </DialogHeader>
          {selectedDocument && (
            <div className="space-y-4">
              {/* Preview Image */}
              {selectedDocument.preview_image_path && (
                <div className="flex justify-center border-b pb-4">
                  <button
                    onClick={() => handleOpenPreview(selectedDocument)}
                    className="cursor-pointer hover:opacity-90 transition-opacity"
                  >
                    <img
                      src={getPreviewUrl(selectedDocument.document_id)}
                      alt={`Preview of ${selectedDocument.filename}`}
                      className="max-w-full max-h-96 object-contain rounded border"
                      onError={(e) => {
                        // Hide image if it fails to load
                        const target = e.target as HTMLImageElement;
                        target.style.display = 'none';
                      }}
                    />
                  </button>
                </div>
              )}
              
              {/* Metadata Details */}
              <div className="space-y-2 pt-4">
                <div><strong>Document ID:</strong> {selectedDocument.document_id}</div>
                <div><strong>Filename:</strong> {selectedDocument.filename}</div>
                <div><strong>File Size:</strong> {formatFileSize(selectedDocument.file_size)}</div>
                <div><strong>MIME Type:</strong> {formatMimeType(selectedDocument.mime_type)}</div>
                <div><strong>Status:</strong> {selectedDocument.status}</div>
                <div><strong>Chunks Created:</strong> {selectedDocument.chunks_created || 'N/A'}</div>
                <div><strong>Upload Date:</strong> {selectedDocument.upload_timestamp ? formatDate(selectedDocument.upload_timestamp) : 'N/A'}</div>
                <div><strong>Storage Path:</strong> {selectedDocument.storage_path}</div>
                {selectedDocument.preview_image_path && (
                  <div><strong>Preview Path:</strong> {selectedDocument.preview_image_path}</div>
                )}
                {selectedDocument.metadata && (
                  <div>
                    <strong>Metadata:</strong>
                    <pre className="mt-2 p-2 bg-muted rounded text-xs overflow-auto">
                      {JSON.stringify(selectedDocument.metadata, null, 2)}
                    </pre>
                  </div>
                )}
              </div>
            </div>
          )}
          <DialogFooter>
            <Button onClick={() => setShowMetadataDialog(false)}>Close</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <Dialog open={showDeleteDialog} onOpenChange={setShowDeleteDialog}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>
              {selectedDocument ? 'Delete Document' : `Delete ${selectedDocuments.size} Document${selectedDocuments.size !== 1 ? 's' : ''}`}
            </DialogTitle>
            <DialogDescription>
              {selectedDocument ? (
                <>
                  Are you sure you want to delete "{selectedDocument.filename}"? This action cannot be undone.
                </>
              ) : (
                <>
                  Are you sure you want to delete {selectedDocuments.size} document{selectedDocuments.size !== 1 ? 's' : ''}? This action cannot be undone.
                </>
              )}
            </DialogDescription>
          </DialogHeader>
          
          {/* Deletion Locations */}
          <div className="space-y-3 py-4">
            <div className="text-sm font-medium">This will delete from the following locations:</div>
            <ul className="list-disc list-inside space-y-2 text-sm text-muted-foreground">
              <li><strong>Azure AI Search:</strong> All document chunks and embeddings</li>
              <li><strong>Supabase Storage:</strong> Main document file and preview image (if exists)</li>
              <li><strong>Database:</strong> Document metadata record</li>
            </ul>
            
            {/* List of documents to delete */}
            {selectedDocument ? (
              <div className="mt-4 p-3 bg-muted rounded-md">
                <div className="font-medium text-sm">{selectedDocument.filename}</div>
                <div className="text-xs text-muted-foreground mt-1">
                  {formatMimeType(selectedDocument.mime_type)} • {formatFileSize(selectedDocument.file_size)}
                </div>
              </div>
            ) : selectedDocuments.size > 0 ? (
              <div className="mt-4 space-y-2 max-h-48 overflow-y-auto">
                {documents
                  .filter(doc => selectedDocuments.has(doc.document_id))
                  .map(doc => (
                    <div key={doc.document_id} className="p-3 bg-muted rounded-md">
                      <div className="font-medium text-sm">{doc.filename}</div>
                      <div className="text-xs text-muted-foreground mt-1">
                        {formatMimeType(doc.mime_type)} • {formatFileSize(doc.file_size)}
                      </div>
                    </div>
                  ))}
              </div>
            ) : null}
          </div>

          <DialogFooter>
            <Button 
              variant="outline" 
              onClick={() => {
                setShowDeleteDialog(false);
                setSelectedDocument(null);
              }}
              disabled={isDeleting}
            >
              Cancel
            </Button>
            <Button 
              variant="destructive" 
              onClick={handleDelete}
              disabled={isDeleting}
            >
              {isDeleting ? 'Deleting...' : 'Delete'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Preview Lightbox Modal */}
      <Dialog open={showPreviewLightbox} onOpenChange={setShowPreviewLightbox}>
        <DialogContent className="max-w-[95vw] max-h-[90vh] p-4 w-auto inline-block">
          {previewLightboxUrl && (
            <div className="relative flex items-center justify-center">
              <img
                src={previewLightboxUrl}
                alt="Document preview"
                className="max-h-[85vh] max-w-[90vw] w-auto h-auto object-contain rounded"
              />
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}

