"use client";

import React, { useMemo } from "react";
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

export default function DynamicWidget({ data }) {
  // 1. Determine Configuration Source
  const chartConfig = useMemo(() => {
    if (!data) return null;

    // Priority A: Structured Config from Backend (PredictionService)
    if (data.chart_config) {
      return data.chart_config;
    }

    // Priority B: Raw React Code from LLM (Legacy/Chatbot)
    if (data.react_code) {
      try {
        let code = data.react_code;
        if (typeof code === "object" && code !== null) return code;
        if (typeof code === "string") {
          code = code.trim().replace(/^```(json)?/i, "").replace(/```$/, "");
          return JSON.parse(code);
        }
      } catch (e) {
        console.error("Failed to parse chart config:", e);
      }
    }
    return null;
  }, [data]);

  // 2. Determine Data Source
  // Backend sends 'predictions' list, Legacy might send 'data' list
  const chartData = data?.predictions || data?.data || [];

  if (!chartConfig || chartData.length === 0) {
    return (
      <div className="flex h-full flex-col items-center justify-center text-sm text-gray-400 p-4 text-center">
        <p>Waiting for data...</p>
        <span className="text-xs opacity-50">
           {chartConfig ? "Data empty" : "Config missing"}
        </span>
      </div>
    );
  }

  const { type, xAxisKey, series = [], colors = [] } = chartConfig;
  const defaultColors = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"];

  // ... (Rendering logic remains the same, just ensuring data is passed correctly) ...
  
  const tooltipStyle = {
    backgroundColor: "#ffffff",
    border: "1px solid #e5e7eb",
    borderRadius: "8px",
    boxShadow: "0 4px 6px -1px rgba(0, 0, 0, 0.1)",
    color: "#374151",
  };

  const renderChart = () => {
    const commonProps = {
      data: chartData, // <--- Using the resolved data source
      margin: { top: 10, right: 30, left: 0, bottom: 0 },
    };

    const AxisProps = {
      stroke: "#9ca3af",
      style: { fontSize: 12, fontFamily: "sans-serif" },
    };

    // Normalize type
    const chartType = (type || "line").toLowerCase().replace("chart", "");

    if (chartType === "bar") {
      return (
        <BarChart {...commonProps}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" vertical={false} />
          <XAxis dataKey={xAxisKey} {...AxisProps} />
          <YAxis {...AxisProps} />
          <Tooltip contentStyle={tooltipStyle} cursor={{ fill: "#f3f4f6" }} />
          <Legend />
          {series.map((s, i) => (
            <Bar
              key={s.dataKey}
              dataKey={s.dataKey}
              name={s.label || s.dataKey}
              fill={s.color || colors[i] || defaultColors[i % defaultColors.length]}
              radius={[4, 4, 0, 0]}
            />
          ))}
        </BarChart>
      );
    }

    // Default to Line Chart
    return (
      <LineChart {...commonProps}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
        <XAxis dataKey={xAxisKey} {...AxisProps} />
        <YAxis {...AxisProps} />
        <Tooltip contentStyle={tooltipStyle} />
        <Legend wrapperStyle={{ paddingTop: "10px" }} />
        {series.map((s, i) => (
          <Line
            key={s.dataKey}
            type="monotone"
            dataKey={s.dataKey}
            name={s.label || s.dataKey}
            stroke={s.color || colors[i] || defaultColors[i % defaultColors.length]}
            strokeWidth={2}
            dot={{ r: 3, strokeWidth: 2 }}
            activeDot={{ r: 6 }}
          />
        ))}
      </LineChart>
    );
  };

  return (
    <div className="h-full min-h-[200px] w-full font-sans p-2">
      <ResponsiveContainer width="100%" height="100%">
        {renderChart()}
      </ResponsiveContainer>
    </div>
  );
}