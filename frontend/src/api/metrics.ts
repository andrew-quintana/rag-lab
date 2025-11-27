/**
 * API client for fetching metrics from backend
 */

export interface Metrics {
  model_quality_by_version: Record<string, any>;
  retrieval_performance: Record<string, any>;
  prompt_stability: Record<string, any>;
  message?: string;
}

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export async function fetchMetrics(): Promise<Metrics> {
  const response = await fetch(`${API_BASE_URL}/api/metrics`);
  
  if (!response.ok) {
    throw new Error(`Failed to fetch metrics: ${response.statusText}`);
  }
  
  return response.json();
}

