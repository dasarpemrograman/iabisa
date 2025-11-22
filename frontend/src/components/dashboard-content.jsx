"use client";

import { useState, useMemo } from "react";
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
} from "recharts";

// Internal Component Imports
// Pastikan menggunakan @/ jika memungkinkan, atau path relative yang konsisten
import { Grid } from "../app/Grid"; 
import { SortableArea } from "../app/SortableArea";
import { useDashboard } from "@/context/dashboard-context"; // Gunakan alias @
import DynamicWidget from "./dynamic-chart";
import BPJSIndonesiaMap from "./map";

// --- THEME SETUP ---
const dark = createTheme({ palette: { mode: "dark" } });
const light = createTheme({ palette: { mode: "light" } });

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

// [WIDGETS PLACEHOLDERS - SAMA SEPERTI SEBELUMNYA]
function LineChartWidget() {
  return (
    <WidgetBase fill>
      <Box sx={{ flex: 1, width: "100%", height: "100%", p: 1 }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={[]} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
             <CartesianGrid strokeDasharray="3 3" />
             <XAxis dataKey="time" />
             <YAxis />
             <Tooltip />
             <Line type="monotone" dataKey="value" stroke="#8884d8" />
          </LineChart>
        </ResponsiveContainer>
      </Box>
    </WidgetBase>
  );
}

function BarChartWidget() {
    return <WidgetBase>Legacy Bar Chart</WidgetBase>;
}

function TimeWidget() {
    return <WidgetBase>Legacy Time Widget</WidgetBase>;
}

const widgets = {
  linechart: LineChartWidget,
  barchart: BarChartWidget,
  time: TimeWidget,
  map: (props) => <BPJSIndonesiaMap {...props} />,
  "dynamic-chart": DynamicWidget,
};

function Content({ data }) {
  const Widget =
    widgets[data.widget] ??
    (() => <WidgetBase>Unknown Widget: {data.widget}</WidgetBase>);

  return (
    <Box
      sx={{
        display: "flex",
        flexDirection: "column",
        width: "100%",
        height: "100%",
        bgcolor: "background.paper",
        borderRadius: 1,
        overflow: "hidden",
      }}
    >
      {data.title && (
        <Box
          sx={{
            p: 1.5,
            pb: 1,
            fontWeight: "bold",
            fontSize: "0.95rem",
            borderBottom: "1px solid",
            borderColor: "divider",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            bgcolor: "action.hover",
          }}
        >
          <span>{data.title}</span>
        </Box>
      )}
      <Box sx={{ flex: 1, minHeight: 0, position: "relative", p: 0 }}>
        <Widget {...data} type={data.widget} />
      </Box>
    </Box>
  );
}

export default function DashBoardContent() {
  const [darkTheme, setDarkTheme] = useState(false);
  
  // Hook ini sekarang aman karena ada fallback
  const { items, updateGrid, reorderItems } = useDashboard();
  
  const itemIds = useMemo(() => items.map((item) => item.id), [items]);
  const [activeId, setActiveId] = useState(null);

  const sensors = useSensors(
    useSensor(PointerSensor, {
      activationConstraint: { distance: 5 },
    }),
  );

  const handleDragOver = useDebouncedCallback(
    ({ active, over }) => {
      if (!over || active.id === over.id) return;
      reorderItems(active.id, over.id);
    },
    150,
    { leading: true },
  );

  return (
    <ThemeProvider theme={darkTheme ? dark : light}>
      <CssBaseline />
      <Box sx={{ py: 2, px: 2, height: "100%" }}>
        <DndContext
          sensors={sensors}
          collisionDetection={closestCenter}
          onDragStart={(event) => setActiveId(event.active.id)}
          onDragOver={handleDragOver}
          onDragEnd={() => setActiveId(null)}
        >
          <SortableContext items={itemIds} strategy={() => {}}>
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