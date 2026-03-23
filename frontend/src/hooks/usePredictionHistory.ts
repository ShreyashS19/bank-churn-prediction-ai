import { useState, useEffect } from "react";
import { HistoryItem } from "@/components/PredictionHistory";

const STORAGE_KEY = "churn_prediction_history";
const MAX_HISTORY_ITEMS = 10; // Reduced from 20
const MAX_PREVIEW_ROWS = 50; // Store only 50 rows for preview

interface StorageHistoryItem {
  id: string;
  filename: string;
  date: string;
  recordsProcessed: number;
  churnRate: number;
  // Store only limited preview data
  previewData: Array<Record<string, unknown>>;
  hasMoreData: boolean;
}

export const usePredictionHistory = () => {
  const [history, setHistory] = useState<HistoryItem[]>([]);

  useEffect(() => {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      try {
        const storageHistory: StorageHistoryItem[] = JSON.parse(stored);
        // Convert storage format back to full HistoryItem format
        const fullHistory: HistoryItem[] = storageHistory.map(item => ({
          id: item.id,
          filename: item.filename,
          date: item.date,
          recordsProcessed: item.recordsProcessed,
          churnRate: item.churnRate,
          data: item.previewData, // Use preview data
        }));
        setHistory(fullHistory);
      } catch (e) {
        console.error("Failed to parse prediction history:", e);
        // Clear corrupted data
        localStorage.removeItem(STORAGE_KEY);
      }
    }
  }, []);

  const addToHistory = (item: Omit<HistoryItem, "id">) => {
    try {
      const newItem: HistoryItem = {
        ...item,
        id: Date.now().toString(),
      };
      
      const updatedHistory = [newItem, ...history].slice(0, MAX_HISTORY_ITEMS);
      setHistory(updatedHistory);
      
      // Convert to storage format (with limited data)
      try {
        const storageHistory: StorageHistoryItem[] = updatedHistory.map(historyItem => ({
          id: historyItem.id,
          filename: historyItem.filename,
          date: historyItem.date,
          recordsProcessed: historyItem.recordsProcessed,
          churnRate: historyItem.churnRate,
          previewData: historyItem.data.slice(0, MAX_PREVIEW_ROWS),
          hasMoreData: historyItem.data.length > MAX_PREVIEW_ROWS,
        }));
        
        localStorage.setItem(STORAGE_KEY, JSON.stringify(storageHistory));
      } catch (storageError) {
        // If storage fails due to quota, remove oldest items and retry
        console.warn("Storage quota exceeded, trimming history...", storageError);
        
        // Try with fewer items
        const trimmedHistory = updatedHistory.slice(0, 5);
        setHistory(trimmedHistory);
        
        try {
          const storageHistory: StorageHistoryItem[] = trimmedHistory.map(historyItem => ({
            id: historyItem.id,
            filename: historyItem.filename,
            date: historyItem.date,
            recordsProcessed: historyItem.recordsProcessed,
            churnRate: historyItem.churnRate,
            previewData: historyItem.data.slice(0, MAX_PREVIEW_ROWS),
            hasMoreData: historyItem.data.length > MAX_PREVIEW_ROWS,
          }));
          
          localStorage.setItem(STORAGE_KEY, JSON.stringify(storageHistory));
        } catch (retryError) {
          // If still failing, clear all history
          console.error("Unable to save history even after trimming:", retryError);
          localStorage.removeItem(STORAGE_KEY);
        }
      }
      
      return newItem;
    } catch (error) {
      console.error("Error adding to history:", error);
      return {
        ...item,
        id: Date.now().toString(),
      };
    }
  };

  const deleteFromHistory = (id: string) => {
    try {
      const updatedHistory = history.filter((item) => item.id !== id);
      setHistory(updatedHistory);
      
      try {
        const storageHistory: StorageHistoryItem[] = updatedHistory.map(historyItem => ({
          id: historyItem.id,
          filename: historyItem.filename,
          date: historyItem.date,
          recordsProcessed: historyItem.recordsProcessed,
          churnRate: historyItem.churnRate,
          previewData: historyItem.data.slice(0, MAX_PREVIEW_ROWS),
          hasMoreData: historyItem.data.length > MAX_PREVIEW_ROWS,
        }));
        
        localStorage.setItem(STORAGE_KEY, JSON.stringify(storageHistory));
      } catch (storageError) {
        console.error("Error updating localStorage after deletion:", storageError);
      }
    } catch (error) {
      console.error("Error deleting from history:", error);
    }
  };

  const clearHistory = () => {
    setHistory([]);
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch (error) {
      console.error("Error clearing history:", error);
    }
  };

  // Helper to get storage size
  const getStorageSize = (): number => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      return stored ? new Blob([stored]).size : 0;
    } catch {
      return 0;
    }
  };

  return {
    history,
    addToHistory,
    deleteFromHistory,
    clearHistory,
    getStorageSize, // Optional: for debugging
  };
};
