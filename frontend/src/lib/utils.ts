/**
 * Utility functions
 */

import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

/**
 * Format MIME type to user-friendly format
 * e.g., "application/pdf" -> "PDF", "image/jpeg" -> "JPEG"
 */
export function formatMimeType(mimeType?: string): string {
  if (!mimeType) return 'Unknown';
  
  // Handle common MIME types
  const mimeTypeMap: Record<string, string> = {
    'application/pdf': 'PDF',
    'image/jpeg': 'JPEG',
    'image/jpg': 'JPEG',
    'image/png': 'PNG',
    'image/gif': 'GIF',
    'image/webp': 'WebP',
    'image/svg+xml': 'SVG',
    'text/plain': 'Text',
    'text/csv': 'CSV',
    'application/msword': 'Word',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'Word',
    'application/vnd.ms-excel': 'Excel',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'Excel',
    'application/vnd.ms-powerpoint': 'PowerPoint',
    'application/vnd.openxmlformats-officedocument.presentationml.presentation': 'PowerPoint',
  };
  
  // Check if we have a direct mapping
  if (mimeTypeMap[mimeType]) {
    return mimeTypeMap[mimeType];
  }
  
  // Extract subtype from MIME type (e.g., "application/pdf" -> "pdf")
  const parts = mimeType.split('/');
  if (parts.length === 2) {
    const subtype = parts[1];
    // Capitalize first letter
    return subtype.charAt(0).toUpperCase() + subtype.slice(1).toUpperCase();
  }
  
  return mimeType;
}

export function formatDate(date: string | Date): string {
  const d = typeof date === 'string' ? new Date(date) : date;
  return d.toLocaleString();
}

export function formatNumber(num: number, decimals: number = 2): string {
  return num.toFixed(decimals);
}

