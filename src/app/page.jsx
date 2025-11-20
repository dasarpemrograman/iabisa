"use client";
import { useEffect, useMemo, useState } from "react";
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
import { arraySwap, SortableContext } from "@dnd-kit/sortable";
import { useDebouncedCallback } from "use-debounce";

import { Grid } from "./Grid";
import { SortableArea } from "./SortableArea";
import { Area } from "./Area";
import { nanoid } from "nanoid";

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
import BPJSIndonesiaMap from "../components/map";

const dark = createTheme({
  palette: {
    mode: "dark",
  },
});

const light = createTheme({
  palette: {
    mode: "light",
  },
});

// sample data for charts
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

// Keep some of the original simple widgets for variety
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

const widgets = {
  linechart: LineChartWidget,
  barchart: BarChartWidget,
  // map widget - wrap the imported map component so it can receive the `data` prop if needed
  map: (props) => <BPJSIndonesiaMap {...props} />,
  time: TimeWidget,
};

function Content({ data }) {
  const Widget =
    widgets[data.widget] ??
    (() => <WidgetBase>{JSON.stringify(data)}</WidgetBase>);
  return <Widget data={data} />;
}

const initialState = [
  {
    id: "chart-1",
    widget: "linechart",
    bgcolor: "#ffffff",
    grid: {
      colSpan: 8,
      rowSpan: 10,
    },
  },
  {
    id: "chart-2",
    widget: "barchart",
    bgcolor: "#ffffff",
    grid: {
      colSpan: 4,
      rowSpan: 10,
    },
  },
  {
    id: "map-1",
    widget: "map",
    bgcolor: "#ffffff",
    grid: {
      // make the map a larger item by default
      colSpan: 8,
      rowSpan: 12,
    },
    // optional: you can provide data through this field if BPJSIndonesiaMap is adapted to accept it
    data: {},
  },
  {
    id: "time-1",
    widget: "time",
    bgcolor: "#f3f3f3",
    grid: {
      colSpan: 4,
      rowSpan: 4,
    },
  },
];

export default function App() {
  const [darkTheme, setDarkTheme] = useState(false);
  const [items, setItems] = useState(initialState);
  const itemIds = useMemo(() => items.map((item) => item.id), [items]);
  const [activeId, setActiveId] = useState(null);

  const sensors = useSensors(
    useSensor(PointerSensor, {
      // default
    }),
  );

  // debounced reordering on drag over to avoid too many swaps
  const handleDragOver = useDebouncedCallback(
    ({ active, over }) => {
      if (!over || active.id === over.id) return;
      setItems((previousState) => {
        const oldIndex = previousState.findIndex(
          (item) => item.id === active.id,
        );
        const newIndex = previousState.findIndex((item) => item.id === over.id);
        if (oldIndex === -1 || newIndex === -1) return previousState;
        return arraySwap(previousState, oldIndex, newIndex);
      });
    },
    150,
    { leading: true },
  );

  return (
    <ThemeProvider theme={darkTheme ? dark : light}>
      <CssBaseline />
      <Box sx={{ py: 2, px: { xs: 1, sm: 2, md: 3, lg: 4 } }}>
        <Box sx={{ mb: 2, display: "flex", gap: 2, alignItems: "center" }}>
          <Box component="h3" sx={{ m: 0 }}>
            Dashboard — Recharts examples
          </Box>
          <Box sx={{ ml: "auto", display: "flex", gap: 1 }}>
            <button
              onClick={() => setDarkTheme((d) => !d)}
              style={{ padding: "6px 10px", cursor: "pointer" }}
            >
              Toggle theme
            </button>
            <button
              onClick={() =>
                setItems((prev) => [
                  ...prev,
                  {
                    id: `chart-${nanoid()}`,
                    widget: "linechart",
                    bgcolor: "#fff",
                    grid: { colSpan: 4, rowSpan: 6 },
                  },
                ])
              }
              style={{ padding: "6px 10px", cursor: "pointer" }}
            >
              Add Line Chart
            </button>
          </Box>
        </Box>

        <DndContext
          sensors={sensors}
          collisionDetection={closestCenter}
          onDragStart={(event) => setActiveId(event.active.id)}
          onDragOver={(event) => handleDragOver(event)}
          onDragEnd={() => setActiveId(null)}
          onDragCancel={() => setActiveId(null)}
        >
          <SortableContext items={itemIds} strategy={() => {}}>
            <Grid columns={4} gap={2}>
              {items.map((props, index) => (
                <SortableArea
                  key={props.id}
                  {...props}
                  index={index}
                  onGridChange={(grid) => {
                    setItems((previousState) =>
                      previousState.map((previousItem) =>
                        previousItem.id === props.id
                          ? { ...previousItem, grid }
                          : previousItem,
                      ),
                    );
                  }}
                >
                  <Content data={props} />
                </SortableArea>
              ))}
            </Grid>
          </SortableContext>

          {/* Drag overlay shows the active item while dragging */}
          {/* Keep overlay simple and re-use Area for consistency */}
          {/* (activeId logic handled above) */}
        </DndContext>
      </Box>
    </ThemeProvider>
  );
}
