// frontend/src/context/dashboard-context.tsx
"use client";

import React, { createContext, useContext, useState, useCallback } from "react";
import { arraySwap } from "@dnd-kit/sortable";
import { nanoid } from "nanoid";

// Define the shape of a dashboard item
export interface DashboardItem {
  id: string;
  widget: "linechart" | "barchart" | "map" | "time" | "dynamic-chart";
  bgcolor: string;
  grid: {
    colSpan: number;
    rowSpan: number;
  };
  data?: any; // To store the dynamic data from the AI
  title?: string;
}

interface DashboardContextType {
  items: DashboardItem[];
  addItem: (item: Omit<DashboardItem, "id">) => void;
  removeItem: (id: string) => void;
  updateGrid: (id: string, grid: { colSpan: number; rowSpan: number }) => void;
  reorderItems: (activeId: string, overId: string) => void;
}

const DashboardContext = createContext<DashboardContextType | undefined>(
  undefined,
);

// Initial default items
const defaultItems: DashboardItem[] = [
  {
    id: "chart-1",
    widget: "linechart",
    bgcolor: "#ffffff",
    grid: { colSpan: 8, rowSpan: 10 },
  },
  {
    id: "chart-2",
    widget: "barchart",
    bgcolor: "#ffffff",
    grid: { colSpan: 4, rowSpan: 10 },
  },
  {
    id: "time-1",
    widget: "time",
    bgcolor: "#f3f3f3",
    grid: { colSpan: 4, rowSpan: 4 },
  },
];

export function DashboardProvider({ children }: { children: React.ReactNode }) {
  const [items, setItems] = useState<DashboardItem[]>(defaultItems);

  const addItem = useCallback((newItem: Omit<DashboardItem, "id">) => {
    setItems((prev) => [...prev, { ...newItem, id: `widget-${nanoid()}` }]);
  }, []);

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
  if (!context) {
    throw new Error("useDashboard must be used within a DashboardProvider");
  }
  return context;
}
