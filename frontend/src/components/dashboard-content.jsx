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
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
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

// --- COLORS ---
const COLORS = ["#10B981", "#34D399", "#6EE7B7", "#065F46", "#A7F3D0"];

// --- DUMMY DATA FOR TEMPLATES ---
const trendData = [
  { name: "Jan", value: 4000, prediction: 2400 },
  { name: "Feb", value: 3000, prediction: 1398 },
  { name: "Mar", value: 2000, prediction: 9800 },
  { name: "Apr", value: 2780, prediction: 3908 },
  { name: "May", value: 1890, prediction: 4800 },
  { name: "Jun", value: 2390, prediction: 3800 },
  { name: "Jul", value: 3490, prediction: 4300 },
];

const categoryData = [
  { name: "Rawat Inap", value: 400 },
  { name: "Rawat Jalan", value: 300 },
  { name: "IGD", value: 300 },
  { name: "MCU", value: 200 },
];

const barData = [
  { name: "Puskesmas", visits: 4000 },
  { name: "Klinik", visits: 3000 },
  { name: "RS Tipe C", visits: 2000 },
  { name: "RS Tipe B", visits: 2780 },
  { name: "RS Tipe A", visits: 1890 },
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
        pt: fill || scroll ? 0 : 1,
        pb: fill || scroll ? 0 : 1,
        px: fill ? 0 : 1,
        flex: 1,
        height: "100%",
      }}
    >
      {children}
    </Box>
  );
}

// 1. Modern Area Chart (Trend)
function TrendWidget() {
  return (
    <WidgetBase fill>
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart
          data={trendData}
          margin={{ top: 10, right: 10, left: -20, bottom: 0 }}
        >
          <defs>
            <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#10B981" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#10B981" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f3f4f6" />
          <XAxis dataKey="name" tick={{ fontSize: 10 }} axisLine={false} tickLine={false} dy={5} />
          <YAxis tick={{ fontSize: 10 }} axisLine={false} tickLine={false} />
          <Tooltip
            contentStyle={{ fontSize: "12px", borderRadius: "8px", border: "none", boxShadow: "0 4px 6px -1px rgba(0,0,0,0.1)" }}
          />
          <Area
            type="monotone"
            dataKey="value"
            stroke="#10B981"
            fillOpacity={1}
            fill="url(#colorValue)"
            strokeWidth={2}
          />
        </AreaChart>
      </ResponsiveContainer>
    </WidgetBase>
  );
}

// 2. Modern Bar Chart (Facilities)
function FacilityBarWidget() {
  return (
    <WidgetBase fill>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={barData}
          margin={{ top: 10, right: 10, left: -20, bottom: 0 }}
          barSize={30}
        >
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f3f4f6" />
          <XAxis dataKey="name" tick={{ fontSize: 10 }} axisLine={false} tickLine={false} dy={5} />
          <YAxis tick={{ fontSize: 10 }} axisLine={false} tickLine={false} />
          <Tooltip
            cursor={{ fill: "#f9fafb" }}
            contentStyle={{ fontSize: "12px", borderRadius: "8px", border: "none", boxShadow: "0 4px 6px -1px rgba(0,0,0,0.1)" }}
          />
          <Bar dataKey="visits" fill="#34D399" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </WidgetBase>
  );
}

// 3. Modern Pie Chart (Donut)
function CategoryPieWidget() {
  return (
    <WidgetBase fill>
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie
            data={categoryData}
            cx="50%"
            cy="50%"
            innerRadius={60}
            outerRadius={80}
            fill="#8884d8"
            paddingAngle={5}
            dataKey="value"
          >
            {categoryData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
            ))}
          </Pie>
          <Tooltip
             contentStyle={{ fontSize: "12px", borderRadius: "8px", border: "none", boxShadow: "0 4px 6px -1px rgba(0,0,0,0.1)" }}
          />
          <Legend
            verticalAlign="bottom"
            height={36}
            iconType="circle"
            iconSize={8}
            wrapperStyle={{ fontSize: "11px" }}
          />
        </PieChart>
      </ResponsiveContainer>
    </WidgetBase>
  );
}

// --- WIDGET REGISTRY ---
// PENTING: Daftarkan ID widget baru di sini agar tidak muncul error "Unknown Widget"
const widgets = {
  "trend-chart": TrendWidget,
  "facility-bar": FacilityBarWidget,
  "category-pie": CategoryPieWidget,
  "map": (props) => <BPJSIndonesiaMap {...props} />, 
  "dynamic-chart": DynamicWidget,
  // Legacy mapping (jika diperlukan untuk backward compatibility)
  "linechart": TrendWidget, 
  "barchart": FacilityBarWidget,
};

// --- CONTENT COMPONENT ---
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
        borderRadius: "12px",
        overflow: "hidden",
        boxShadow: "0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)",
        border: "1px solid",
        borderColor: "#e5e7eb",
      }}
    >
      {/* Header Title */}
      {data.title && (
        <Box
          sx={{
            px: 2,
            py: 1.5,
            fontWeight: "600",
            fontSize: "0.875rem",
            color: "#111827",
            borderBottom: "1px solid",
            borderColor: "#f3f4f6",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            bgcolor: "white",
          }}
        >
          <span>{data.title}</span>
        </Box>
      )}

      {/* Widget Content */}
      <Box sx={{ flex: 1, minHeight: 0, position: "relative", p: 1, bgcolor: "white" }}>
        <Widget {...data} type={data.widget} />
      </Box>
    </Box>
  );
}

// --- MAIN EXPORT ---
export default function DashBoardContent() {
  const [darkTheme, setDarkTheme] = useState(false);
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
      <Box sx={{ py: 3, px: 3, height: "100%", bgcolor: "#F9FAFB" }}>
        <DndContext
          sensors={sensors}
          collisionDetection={closestCenter}
          onDragStart={(event) => setActiveId(event.active.id)}
          onDragOver={handleDragOver}
          onDragEnd={() => setActiveId(null)}
        >
          <SortableContext items={itemIds} strategy={() => {}}>
            {/* FIX: gap dikembalikan ke 2 (standard) */}
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