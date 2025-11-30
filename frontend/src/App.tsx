import { useState } from 'react';
import { MetricsDashboard } from './components/MetricsDashboard';
import { FileUpload } from './components/FileUpload';
import { FileBrowser } from './components/FileBrowser';
import { Home } from './components/Home';
import { Button } from './components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from './components/ui/card';
import './App.css';

type View = 'home' | 'documents' | 'metrics';

function App() {
  const [activeView, setActiveView] = useState<View>('home');
  const [refreshTrigger, setRefreshTrigger] = useState(0);

  const handleUploadSuccess = () => {
    setRefreshTrigger(prev => prev + 1);
  };

  const handleNavigate = (view: 'documents' | 'metrics') => {
    setActiveView(view);
  };

  return (
    <div className="min-h-screen bg-background p-8">
      <div className="max-w-7xl mx-auto space-y-8">
        <div className="flex items-center justify-between">
          <button
            onClick={() => setActiveView('home')}
            className="text-3xl font-bold hover:text-primary transition-colors cursor-pointer"
          >
            RAG Lab
          </button>
          <div className="flex gap-2">
            <Button
              variant={activeView === 'documents' ? 'default' : 'outline'}
              onClick={() => setActiveView('documents')}
            >
              Documents
            </Button>
            <Button
              variant={activeView === 'metrics' ? 'default' : 'outline'}
              onClick={() => setActiveView('metrics')}
            >
              Metrics
            </Button>
          </div>
        </div>

        {activeView === 'home' && (
          <Home onNavigate={handleNavigate} />
        )}

        {activeView === 'documents' && (
          <div className="space-y-8">
            <Card>
              <CardHeader>
                <CardTitle>Upload Document</CardTitle>
              </CardHeader>
              <CardContent>
                <FileUpload
                  onUploadSuccess={handleUploadSuccess}
                  onUploadError={(error) => {
                    console.error('Upload error:', error);
                  }}
                />
              </CardContent>
            </Card>

            <FileBrowser refreshTrigger={refreshTrigger} />
          </div>
        )}

        {activeView === 'metrics' && (
          <MetricsDashboard />
        )}
      </div>
    </div>
  );
}

export default App
