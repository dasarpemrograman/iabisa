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
  const chartConfig = useMemo(() => {
    if (!data || !data.react_code) return null;

    try {
      let code = data.react_code;

      // 1. Handle if it's already an object
      if (typeof code === "object" && code !== null) return code;

      // 2. Clean string input (Remove Markdown code blocks)
      if (typeof code === "string") {
        code = code.trim();
        // Remove ```json and ``` wrappers if present
        if (code.startsWith("```")) {
          code = code.replace(/^```(json)?/i, "").replace(/```$/, "");
        }
        return JSON.parse(code);
      }

      return null;
    } catch (e) {
      console.error("Failed to parse chart config:", e);
      return null;
    }
  }, [data]);

  if (!chartConfig || !data.data) {
    return (
      <div className="flex h-full animate-pulse items-center justify-center text-sm text-gray-400">
        Waiting for visualization config...
      </div>
    );
  }

  const { data: chartData } = data;
  const { type, xAxisKey, series = [], colors = [] } = chartConfig;
  const defaultColors = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"];

  // Styles for Light Mode
  const tooltipStyle = {
    backgroundColor: "#ffffff",
    border: "1px solid #e5e7eb",
    borderRadius: "8px",
    boxShadow: "0 4px 6px -1px rgba(0, 0, 0, 0.1)",
    color: "#374151",
  };

  const renderChart = () => {
    const commonProps = {
      data: chartData,
      margin: { top: 10, right: 30, left: 0, bottom: 0 },
    };

    const AxisProps = {
      stroke: "#9ca3af",
      style: { fontSize: 12, fontFamily: "sans-serif" },
    };

    // Normalize type to lowercase to handle "LineChart" vs "linechart"
    const chartType = type?.toLowerCase() || "linechart";

    switch (chartType) {
      case "linechart":
      case "line":
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
                stroke={
                  s.color ||
                  colors[i] ||
                  defaultColors[i % defaultColors.length]
                }
                strokeWidth={2}
                dot={{ r: 3, strokeWidth: 2 }}
                activeDot={{ r: 6 }}
              />
            ))}
          </LineChart>
        );

      case "barchart":
      case "bar":
        return (
          <BarChart {...commonProps}>
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="#e5e7eb"
              vertical={false}
            />
            <XAxis dataKey={xAxisKey} {...AxisProps} />
            <YAxis {...AxisProps} />
            <Tooltip contentStyle={tooltipStyle} cursor={{ fill: "#f3f4f6" }} />
            <Legend />
            {series.map((s, i) => (
              <Bar
                key={s.dataKey}
                dataKey={s.dataKey}
                name={s.label || s.dataKey}
                fill={
                  s.color ||
                  colors[i] ||
                  defaultColors[i % defaultColors.length]
                }
                radius={[4, 4, 0, 0]}
              />
            ))}
          </BarChart>
        );

      case "areachart":
      case "area":
        return (
          <AreaChart {...commonProps}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis dataKey={xAxisKey} {...AxisProps} />
            <YAxis {...AxisProps} />
            <Tooltip contentStyle={tooltipStyle} />
            <Legend />
            {series.map((s, i) => (
              <Area
                key={s.dataKey}
                type="monotone"
                dataKey={s.dataKey}
                name={s.label || s.dataKey}
                stroke={
                  s.color ||
                  colors[i] ||
                  defaultColors[i % defaultColors.length]
                }
                fill={
                  s.color ||
                  colors[i] ||
                  defaultColors[i % defaultColors.length]
                }
                fillOpacity={0.3}
              />
            ))}
          </AreaChart>
        );

      default:
        return (
          <div className="flex h-full items-center justify-center text-sm text-red-400">
            Unsupported chart type: {type}
          </div>
        );
    }
  };

  return (
    <div className="h-full min-h-[200px] w-full font-sans">
      <ResponsiveContainer width="100%" height="100%">
        {renderChart()}
      </ResponsiveContainer>
    </div>
  );
}
