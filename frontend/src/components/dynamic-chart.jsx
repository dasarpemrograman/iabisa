// src/components/dynamic-chart.jsx
"use client";

import {
  LineChart,
  Line,
  BarChart,
  Bar,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import BPJSIndonesiaMap from "./map"; // Assuming this exists from your uploaded files
import { Box, Typography } from "@mui/material";

export default function DynamicWidget({ data, type, title }) {
  // Heuristic: Detect visualization type if not explicitly provided
  // The backend sends 'component_name' which usually implies type

  const isMap = type === "map" || title?.toLowerCase().includes("map");

  if (isMap) {
    return (
      <Box sx={{ width: "100%", height: "100%", minHeight: 300, p: 1 }}>
        {title && (
          <Typography variant="subtitle2" mb={1}>
            {title}
          </Typography>
        )}
        <BPJSIndonesiaMap data={data} />
      </Box>
    );
  }

  // Chart Logic
  const keys = data && data.length > 0 ? Object.keys(data[0]) : [];
  const xAxisKey =
    keys.find(
      (k) =>
        k.toLowerCase().includes("date") ||
        k.toLowerCase().includes("time") ||
        k.toLowerCase().includes("name") ||
        k.toLowerCase().includes("province"),
    ) || keys[0];
  const dataKey =
    keys.find((k) => k !== xAxisKey && typeof data[0][k] === "number") ||
    keys[1];

  const isBar =
    type === "barchart" ||
    title?.toLowerCase().includes("bar") ||
    keys.length < 3; // Simple heuristic

  return (
    <Box sx={{ width: "100%", height: "100%", minHeight: 250, p: 1 }}>
      {title && (
        <Typography variant="subtitle2" mb={1}>
          {title}
        </Typography>
      )}
      <ResponsiveContainer width="100%" height="100%">
        {isBar ? (
          <BarChart
            data={data}
            margin={{ top: 10, right: 10, left: 0, bottom: 0 }}
          >
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey={xAxisKey} style={{ fontSize: 12 }} />
            <YAxis style={{ fontSize: 12 }} />
            <Tooltip
              contentStyle={{
                borderRadius: 8,
                border: "none",
                boxShadow: "0 4px 12px rgba(0,0,0,0.1)",
              }}
            />
            <Legend />
            <Bar dataKey={dataKey} fill="#2A4491" radius={[4, 4, 0, 0]} />
          </BarChart>
        ) : (
          <LineChart
            data={data}
            margin={{ top: 10, right: 10, left: 0, bottom: 0 }}
          >
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey={xAxisKey} style={{ fontSize: 12 }} />
            <YAxis style={{ fontSize: 12 }} />
            <Tooltip
              contentStyle={{
                borderRadius: 8,
                border: "none",
                boxShadow: "0 4px 12px rgba(0,0,0,0.1)",
              }}
            />
            <Line
              type="monotone"
              dataKey={dataKey}
              stroke="#44853B"
              strokeWidth={3}
              dot={{ r: 4, strokeWidth: 2 }}
              activeDot={{ r: 6 }}
            />
          </LineChart>
        )}
      </ResponsiveContainer>
    </Box>
  );
}
