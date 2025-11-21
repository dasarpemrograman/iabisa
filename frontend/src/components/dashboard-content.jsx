"use client";

import { useState, useMemo, useEffect } from "react";
import Box from "@mui/material/Box";
import CssBaseline from "@mui/material/CssBaseline";
import { ThemeProvider, createTheme } from "@mui/material/styles";
import {
  DndContext,
  closestCenter,
  PointerSensor,
  useSensor,
  useSensors,
} from "@dnd-kit/core";
import { SortableContext } from "@dnd-kit/sortable";
import { useDebouncedCallback } from "use-debounce";
import {
  LineChart,
  Line,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  Legend,
} from "recharts";

// Internal Component Imports
import { Grid } from "../app/Grid";
import { SortableArea } from "../app/SortableArea";
import { useDashboard } from "@/context/dashboard-context";
import DynamicWidget from "./dynamic-chart";
import BPJSIndonesiaMap from "./map";

// --- THEME SETUP ---
const dark = createTheme({ palette: { mode: "dark" } });
const light = createTheme({ palette: { mode: "light" } });

// --- SAMPLE DATA (Legacy) ---
const hourlyData = [
  { time: "00:00", value: 40 },
  { time: "02:00", value: 30 },
  { time: "04:00", value: 20 },
  { time: "06:00", value: 27 },
  { time: "08:00", value: 50 },
  { time: "10:00", value: 65 },
  { time: "12:00", value: 80 },
  { time: "14:00", value: 75 },
  { time: "16:00", value: 60 },
  { time: "18:00", value: 55 },
  { time: "20:00", value: 48 },
  { time: "22:00", value: 42 },
];

const categoryData = [
  { name: "A", count: 30 },
  { name: "B", count: 55 },
  { name: "C", count: 20 },
  { name: "D", count: 75 },
  { name: "E", count: 45 },
];

// --- WIDGET COMPONENTS ---

function WidgetBase({ children, fill, scroll, ...props }) {
  return (
    <Box
      {...props}
      sx={{
        display: "flex",
        flexDirection: "column",
        overflow: scroll ? "auto" : undefined,
        pt: fill || scroll ? 0 : 2,
        pb: fill || scroll ? 0 : 3,
        px: fill ? 0 : 2,
        flex: 1,
      }}
    >
      {children}
    </Box>
  );
}

function LineChartWidget() {
  return (
    <WidgetBase fill>
      <Box sx={{ flex: 1, width: "100%", height: "100%", p: 1 }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={hourlyData}
            margin={{ top: 8, right: 16, left: 0, bottom: 8 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="time" />
            <YAxis />
            <Tooltip />
            <Line
              type="monotone"
              dataKey="value"
              stroke="#8884d8"
              strokeWidth={2}
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </Box>
    </WidgetBase>
  );
}

function BarChartWidget() {
  return (
    <WidgetBase fill>
      <Box sx={{ flex: 1, width: "100%", height: "100%", p: 1 }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={categoryData}
            margin={{ top: 8, right: 16, left: 0, bottom: 8 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="count" fill="#82ca9d" />
          </BarChart>
        </ResponsiveContainer>
      </Box>
    </WidgetBase>
  );
}

function TimeWidget() {
  const [time, setTime] = useState(null);
  useEffect(() => {
    setTime(new Date().toLocaleTimeString());
    const interval = setInterval(() => {
      setTime(new Date().toLocaleTimeString());
    }, 1000);
    return () => clearInterval(interval);
  }, []);
  return (
    <WidgetBase fill>
      <Box
        component="pre"
        sx={{
          flex: 1,
          fontSize: "2.2em",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          textAlign: "center",
          p: 0,
          m: 0,
        }}
      >
        {time ?? "--:--:--"}
      </Box>
    </WidgetBase>
  );
}

// --- WIDGET REGISTRY ---
const widgets = {
  linechart: LineChartWidget,
  barchart: BarChartWidget,
  time: TimeWidget,
  // Map uses a wrapper to ensure it handles props correctly
  map: (props) => <BPJSIndonesiaMap {...props} />,
  // Dynamic chart handles AI output
  "dynamic-chart": DynamicWidget,
};

function Content({ data }) {
  const Widget =
    widgets[data.widget] ??
    (() => <WidgetBase>Unknown Widget: {data.widget}</WidgetBase>);

  // Pass specific props for DynamicWidget (data, type, title)
  // Pass generic props for legacy widgets
  return <Widget {...data} type={data.widget} />;
}

// --- MAIN EXPORT ---

export default function DashBoardContent() {
  const [darkTheme, setDarkTheme] = useState(false);

  // Access global state from Context
  const { items, updateGrid, reorderItems } = useDashboard();

  const itemIds = useMemo(() => items.map((item) => item.id), [items]);
  const [activeId, setActiveId] = useState(null);

  const sensors = useSensors(
    useSensor(PointerSensor, {
      activationConstraint: { distance: 5 },
    }),
  );

  // Debounced reordering to prevent flicker
  const handleDragOver = useDebouncedCallback(
    ({ active, over }) => {
      if (!over || active.id === over.id) return;
      // Call context function to swap items
      reorderItems(active.id, over.id);
    },
    150,
    { leading: true },
  );

  return (
    <ThemeProvider theme={darkTheme ? dark : light}>
      <CssBaseline />
      <Box sx={{ py: 2, px: { xs: 1, sm: 2, md: 3, lg: 4 }, height: "100%" }}>
        <DndContext
          sensors={sensors}
          collisionDetection={closestCenter}
          onDragStart={(event) => setActiveId(event.active.id)}
          onDragOver={handleDragOver}
          onDragEnd={() => setActiveId(null)}
        >
          <SortableContext items={itemIds} strategy={() => {}}>
            {/* CHANGED: Gap reduced from 10 to 2 */}
            <Grid columns={12} gap={2}>
              {items.map((props, index) => (
                <SortableArea
                  key={props.id}
                  {...props}
                  index={index}
                  onGridChange={(grid) => updateGrid(props.id, grid)}
                >
                  <Content data={props} />
                </SortableArea>
              ))}
            </Grid>
          </SortableContext>
        </DndContext>
      </Box>
    </ThemeProvider>
  );
}
