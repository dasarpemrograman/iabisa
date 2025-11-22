"use client";

import React, { createContext, useContext, useState, useCallback } from "react";
import { arraySwap } from "@dnd-kit/sortable";

// Define the shape of a dashboard item
export interface DashboardItem {
  id: string;
  widget: string;
  bgcolor?: string;
  grid: {
    colSpan: number;
    rowSpan: number;
  };
  data?: any;
  title?: string;
}

interface DashboardContextType {
  items: DashboardItem[];
  addItem: (widgetType: string, title: string, data: any) => void;
  removeItem: (id: string) => void;
  updateGrid: (id: string, grid: { colSpan: number; rowSpan: number }) => void;
  reorderItems: (activeId: string, overId: string) => void;
}

// Inisialisasi dengan null
const DashboardContext = createContext<DashboardContextType | null>(null);

const defaultItems: DashboardItem[] = [
  {
    id: "trend-1",
    widget: "trend-chart", // Menggunakan widget Area Chart baru
    bgcolor: "#ffffff",
    grid: { colSpan: 8, rowSpan: 8 }, // Lebar (2/3 layar)
    title: "Pertumbuhan Peserta JKN (2024)",
  },
  {
    id: "pie-1",
    widget: "category-pie", // Menggunakan widget Pie Chart baru
    bgcolor: "#ffffff",
    grid: { colSpan: 4, rowSpan: 8 }, // Sempit (1/3 layar)
    title: "Distribusi Layanan Faskes",
  },
  {
    id: "bar-1",
    widget: "facility-bar", // Menggunakan widget Bar Chart baru
    bgcolor: "#ffffff",
    grid: { colSpan: 6, rowSpan: 8 }, // Setengah layar
    title: "Kunjungan per Tipe Faskes",
  },
  {
    id: "map-1",
    widget: "map", // Map Widget
    bgcolor: "#ffffff",
    grid: { colSpan: 6, rowSpan: 8 }, // Setengah layar
    title: "Peta Sebaran Faskes Indonesia",
  },
];

export function DashboardProvider({ children }: { children: React.ReactNode }) {
  const [items, setItems] = useState<DashboardItem[]>(defaultItems);

  const addItem = useCallback(
    (widgetType: string, title: string, data: any) => {
      setItems((prev) => {
        // Gunakan crypto.randomUUID() atau fallback sederhana untuk SSR safe
        const id = typeof crypto !== 'undefined' && crypto.randomUUID 
          ? crypto.randomUUID() 
          : `widget-${Math.random().toString(36).substr(2, 9)}`;

        const newItem: DashboardItem = {
          id: `widget-${id}`,
          widget: widgetType,
          title: title,
          data: data,
          bgcolor: "#ffffff",
          grid: {
            colSpan: 6,
            rowSpan: 8,
          },
        };
        return [...prev, newItem];
      });
    },
    [],
  );

  const removeItem = useCallback((id: string) => {
    setItems((prev) => prev.filter((item) => item.id !== id));
  }, []);

  const updateGrid = useCallback(
    (id: string, grid: { colSpan: number; rowSpan: number }) => {
      setItems((prev) =>
        prev.map((item) => (item.id === id ? { ...item, grid } : item)),
      );
    },
    [],
  );

  const reorderItems = useCallback((activeId: string, overId: string) => {
    setItems((prev) => {
      const oldIndex = prev.findIndex((item) => item.id === activeId);
      const newIndex = prev.findIndex((item) => item.id === overId);
      if (oldIndex !== -1 && newIndex !== -1) {
        return arraySwap(prev, oldIndex, newIndex);
      }
      return prev;
    });
  }, []);

  return (
    <DashboardContext.Provider
      value={{ items, addItem, removeItem, updateGrid, reorderItems }}
    >
      {children}
    </DashboardContext.Provider>
  );
}

export function useDashboard() {
  const context = useContext(DashboardContext);
  
  // FIX: Fallback aman untuk mencegah crash saat SSR atau masalah hydration
  if (!context) {
    console.warn("useDashboard dipanggil di luar DashboardProvider. Menggunakan fallback.");
    return {
      items: defaultItems,
      addItem: () => console.warn("addItem: Provider missing"),
      removeItem: () => console.warn("removeItem: Provider missing"),
      updateGrid: () => console.warn("updateGrid: Provider missing"),
      reorderItems: () => console.warn("reorderItems: Provider missing"),
    };
  }
  
  return context;
}