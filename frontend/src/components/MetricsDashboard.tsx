import { useEffect, useState } from 'react';
import { fetchMetrics, Metrics } from '../api/metrics';

export function MetricsDashboard() {
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadMetrics() {
      try {
        const data = await fetchMetrics();
        setMetrics(data);
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load metrics');
      } finally {
        setLoading(false);
      }
    }

    loadMetrics();
    // Refresh every 30 seconds
    const interval = setInterval(loadMetrics, 30000);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return <div>Loading metrics...</div>;
  }

  if (error) {
    return <div>Error: {error}</div>;
  }

  if (!metrics) {
    return <div>No metrics available</div>;
  }

  return (
    <div>
      <h1>RAG Evaluation Platform - Metrics Dashboard</h1>
      <div>
        <h2>Model Quality by Version</h2>
        <pre>{JSON.stringify(metrics.model_quality_by_version, null, 2)}</pre>
      </div>
      <div>
        <h2>Retrieval Performance</h2>
        <pre>{JSON.stringify(metrics.retrieval_performance, null, 2)}</pre>
      </div>
      <div>
        <h2>Prompt Stability</h2>
        <pre>{JSON.stringify(metrics.prompt_stability, null, 2)}</pre>
      </div>
    </div>
  );
}

