import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { BarChart3, Upload } from 'lucide-react';

interface HomeProps {
  onNavigate: (view: 'documents' | 'metrics') => void;
}

export function Home({ onNavigate }: HomeProps) {
  return (
    <div className="space-y-8">
      <Card>
        <CardHeader>
          <CardTitle className="text-2xl">Welcome to RAG Lab</CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <p className="text-muted-foreground text-lg">
            A platform for evaluating and managing Retrieval-Augmented Generation (RAG) systems.
          </p>

          <div className="grid md:grid-cols-2 gap-6 mt-8">
            <Card>
              <CardHeader>
                <div className="flex items-center gap-2">
                  <Upload className="h-5 w-5" />
                  <CardTitle>Document Management</CardTitle>
                </div>
              </CardHeader>
              <CardContent className="space-y-3">
                <p className="text-sm text-muted-foreground">
                  Upload and manage documents for your RAG system:
                </p>
                <ul className="text-sm space-y-2 list-disc list-inside text-muted-foreground">
                  <li>Upload PDFs, images, and other documents</li>
                  <li>Browse and search uploaded documents</li>
                  <li>View document metadata and previews</li>
                  <li>Download or delete documents</li>
                  <li>Filter by type, status, date, and size</li>
                </ul>
                <Button 
                  onClick={() => onNavigate('documents')}
                  className="w-full mt-4"
                >
                  Go to Documents
                </Button>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <div className="flex items-center gap-2">
                  <BarChart3 className="h-5 w-5" />
                  <CardTitle>Metrics & Evaluation</CardTitle>
                </div>
              </CardHeader>
              <CardContent className="space-y-3">
                <p className="text-sm text-muted-foreground">
                  Monitor and evaluate your RAG system performance:
                </p>
                <ul className="text-sm space-y-2 list-disc list-inside text-muted-foreground">
                  <li>View query performance metrics</li>
                  <li>Track retrieval accuracy</li>
                  <li>Monitor model answer quality</li>
                  <li>Analyze evaluation judgments</li>
                  <li>Compare prompt versions</li>
                </ul>
                <Button 
                  onClick={() => onNavigate('metrics')}
                  variant="outline"
                  className="w-full mt-4"
                >
                  Go to Metrics
                </Button>
              </CardContent>
            </Card>
          </div>

          <div className="mt-8 pt-6 border-t">
            <h3 className="font-semibold mb-3">Quick Start</h3>
            <ol className="space-y-2 text-sm text-muted-foreground">
              <li className="flex gap-2">
                <span className="font-semibold text-foreground">1.</span>
                <span>Upload documents using the Documents tab. Supported formats include PDF, JPEG, PNG, and GIF.</span>
              </li>
              <li className="flex gap-2">
                <span className="font-semibold text-foreground">2.</span>
                <span>Documents are automatically processed: text extraction, chunking, embedding generation, and indexing.</span>
              </li>
              <li className="flex gap-2">
                <span className="font-semibold text-foreground">3.</span>
                <span>Use the file browser to search, filter, and manage your uploaded documents.</span>
              </li>
              <li className="flex gap-2">
                <span className="font-semibold text-foreground">4.</span>
                <span>Monitor system performance and evaluation metrics in the Metrics tab.</span>
              </li>
            </ol>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

