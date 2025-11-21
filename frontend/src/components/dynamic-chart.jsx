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
import BPJSIndonesiaMap from "./map";
import { Box, Typography } from "@mui/material";

export default function DynamicWidget({ data, type, title }) {
  // --- DETECTION LOGIC ---
  const isExplicitMap = type === "map" || title?.toLowerCase().includes("map");
  const isMapData = data && !Array.isArray(data) && "province_key" in data;
  const isMap = isExplicitMap || isMapData;

  // --- SHARED FLEX CONTAINER STYLE ---
  // This ensures the widget fits exactly inside the dashboard tile
  const containerStyle = {
    width: "100%",
    height: "100%",
    display: "flex",
    flexDirection: "column",
    p: 1,
    overflow: "hidden", // Prevents spillover
  };

  if (isMap) {
    return (
      <Box sx={containerStyle}>
        {title && (
          <Typography variant="subtitle2" mb={1} sx={{ flexShrink: 0 }}>
            {title}
          </Typography>
        )}
        {/* Map takes all remaining height */}
        <Box sx={{ flexGrow: 1, minHeight: 0, position: "relative" }}>
          <BPJSIndonesiaMap data={data} />
        </Box>
      </Box>
    );
  }

  // --- CHART LOGIC ---
  if (!data || !Array.isArray(data)) {
    return (
      <Box
        sx={{
          ...containerStyle,
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <Typography variant="body2" color="text.secondary">
          No chart data available
        </Typography>
      </Box>
    );
  }

  const keys = data.length > 0 ? Object.keys(data[0]) : [];
  const xAxisKey =
    keys.find((k) =>
      ["date", "time", "name", "province"].some((term) =>
        k.toLowerCase().includes(term),
      ),
    ) || keys[0];
  const dataKey =
    keys.find((k) => k !== xAxisKey && typeof data[0][k] === "number") ||
    keys[1];
  const isBar =
    type === "barchart" ||
    title?.toLowerCase().includes("bar") ||
    keys.length < 3;

  return (
    <Box sx={containerStyle}>
      {title && (
        <Typography variant="subtitle2" mb={1} sx={{ flexShrink: 0 }}>
          {title}
        </Typography>
      )}
      <Box sx={{ flexGrow: 1, minHeight: 0 }}>
        <ResponsiveContainer width="100%" height="100%">
          {isBar ? (
            <BarChart
              data={data}
              margin={{ top: 5, right: 5, left: -20, bottom: 0 }}
            >
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis
                dataKey={xAxisKey}
                style={{ fontSize: 10 }}
                tick={{ fill: "#666" }}
              />
              <YAxis style={{ fontSize: 10 }} tick={{ fill: "#666" }} />
              <Tooltip
                contentStyle={{
                  borderRadius: 8,
                  border: "none",
                  boxShadow: "0 4px 12px rgba(0,0,0,0.1)",
                }}
              />
              <Legend wrapperStyle={{ fontSize: "10px" }} />
              <Bar dataKey={dataKey} fill="#2A4491" radius={[4, 4, 0, 0]} />
            </BarChart>
          ) : (
            <LineChart
              data={data}
              margin={{ top: 5, right: 5, left: -20, bottom: 0 }}
            >
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis
                dataKey={xAxisKey}
                style={{ fontSize: 10 }}
                tick={{ fill: "#666" }}
              />
              <YAxis style={{ fontSize: 10 }} tick={{ fill: "#666" }} />
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
                strokeWidth={2}
                dot={{ r: 3 }}
              />
            </LineChart>
          )}
        </ResponsiveContainer>
      </Box>
    </Box>
  );
}
