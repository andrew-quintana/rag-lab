/**
 * Utility functions
 */

export function formatDate(date: string | Date): string {
  const d = typeof date === 'string' ? new Date(date) : date;
  return d.toLocaleString();
}

export function formatNumber(num: number, decimals: number = 2): string {
  return num.toFixed(decimals);
}

