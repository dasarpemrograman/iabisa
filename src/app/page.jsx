"use client";
import { lazy, useEffect, useMemo, useState } from "react";
import Box from "@mui/material/Box";
import CssBaseline from "@mui/material/CssBaseline";
import AppBar from "@mui/material/AppBar";
import Switch from "@mui/material/Switch";
import {
  DndContext,
  closestCenter,
  KeyboardSensor,
  PointerSensor,
  useSensor,
  useSensors,
  DragOverlay,
} from "@dnd-kit/core";
import {
  arrayMove,
  arraySwap,
  SortableContext,
  sortableKeyboardCoordinates,
} from "@dnd-kit/sortable";
import CardContent from "@mui/material/CardContent";
import CardMedia from "@mui/material/CardMedia";
import FormControlLabel from "@mui/material/FormControlLabel";
import { ThemeProvider, createTheme } from "@mui/material/styles";
import Toolbar from "@mui/material/Toolbar";
import Typography from "@mui/material/Typography";
import { useDebouncedCallback } from "use-debounce";
import photos from "./photos.json";

import { Grid } from "./Grid";
import { SortableArea } from "./SortableArea";
import { Area } from "./Area";
import { nanoid } from "nanoid";

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

function powerRandom() {
  return Math.pow(Math.random(), 2);
}

function sqrtRandom() {
  return Math.sqrt(Math.random());
}

function superRandom(max, min = 0) {
  return Math.round(sqrtRandom() * powerRandom() * (max - min)) + min;
}

function random(max, min = 0) {
  return Math.round(Math.random() * (max - min)) + min;
}

function toHex(n) {
  const hex = n.toString(16);
  return hex.length < 2 ? `${hex}${hex}` : hex;
}
function randomHex() {
  const r = toHex(random(255));
  const g = toHex(random(255));
  const b = toHex(random(255));
  return `#${r}${g}${b}`;
}

function timeString() {
  const date = new Date();
  const ss = date.getSeconds().toString().padStart(2, "0");
  const mm = date.getMinutes().toString().padStart(2, "0");
  const hh = date.getHours().toString().padStart(2, "0");
  return `${hh}:${mm}:${ss}`;
}

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
function TimeWidget() {
  // Render a deterministic placeholder on the server and then populate the real time on the client.
  // This avoids server/client markup mismatches caused by Date() during SSR.
  const [time, setTime] = useState(null);
  useEffect(() => {
    setTime(timeString());
    const interval = setInterval(() => {
      setTime(timeString());
    }, 1000);
    return () => {
      clearInterval(interval);
    };
  }, []);
  return (
    <WidgetBase fill>
      <Box
        component="pre"
        sx={{
          flex: 1,
          fontSize: "3em",
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

function LogWidget({ data }) {
  return (
    <WidgetBase>
      <Box component="pre">{JSON.stringify(data, null, 2)}</Box>
    </WidgetBase>
  );
}

function ImageWidget({ data }) {
  return (
    <WidgetBase fill>
      <Box
        component="img"
        src={data.src}
        alt={data.alt}
        sx={{
          height: "100%",
          width: "100%",
          objectFit: "cover",
          objectPosition: "center",
        }}
      />
    </WidgetBase>
  );
}

const widgets = {
  time: TimeWidget,
  log: LogWidget,
  image: ImageWidget,
};

function Content({ data }) {
  const Widget = widgets[data.widget] ?? widgets.log;
  return <Widget data={data} />;
}

function randomWidget() {
  const keys = Object.keys(widgets);
  const index = random(keys.length);
  const key = keys[index];
  return key;
}

const initialState = [
  {
    // Use deterministic ids and values to avoid SSR/CSR mismatches.
    id: "item-1",
    bgcolor: "#fed012",
    widget: "log",
    grid: {
      colSpan: 8,
      rowSpan: 14,
    },
  },
  {
    id: "item-2",
    src: photos[0],
    alt: "",
    bgcolor: "#012fed",
    widget: "image",
    grid: {
      colSpan: 4,
      rowSpan: 11,
    },
  },
  {
    id: "item-3",
    widget: "time",
    bgcolor: "#1fe23d",
    grid: {
      colSpan: 4,
      rowSpan: 3,
    },
  },
  {
    id: "item-4",
    src: photos[1],
    alt: "",
    bgcolor: "#235acd",
    widget: "image",
    grid: {
      colSpan: 4,
      rowSpan: 3,
    },
  },
  {
    id: "item-5",
    src: photos[2],
    alt: "",
    bgcolor: "#41ad56",
    widget: "image",
    grid: {
      colSpan: 8,
      rowSpan: 3,
    },
  },
];

const App = () => {
  const [darkTheme, setDarkTheme] = useState(false);
  const [items, setItems] = useState(initialState);

  const itemIds = useMemo(() => items.map((item) => item.id), [items]);
  const [activeId, setActiveId] = useState(null);

  const activeItem = useMemo(
    () => items.find((item) => activeId === item.id),
    [activeId, items],
  );

  const sensors = useSensors(
    useSensor(PointerSensor, {
      // Adding delay
      // activationConstraint: {
      //   delay: 100,
      //   tolerance: 10
      // }
    }),
    useSensor(KeyboardSensor, {
      coordinateGetter: sortableKeyboardCoordinates,
    }),
  );
  const handleDragOver = useDebouncedCallback(
    ({ active, over }) => {
      if (active.id !== over.id) {
        setItems((previousState) => {
          const oldIndex = previousState.findIndex(
            (item) => item.id === active.id,
          );
          const newIndex = previousState.findIndex(
            (item) => item.id === over.id,
          );
          // Swap or move ?
          // return arrayMove(previousState, oldIndex, newIndex);
          return arraySwap(previousState, oldIndex, newIndex);
        });
      }
    },
    250,
    { leading: true },
  );
  return (
    <ThemeProvider theme={darkTheme ? dark : light}>
      <CssBaseline />
      <div className="sticky w-full px-2">
        <h3>Mui with dnd-kit</h3>
      </div>
      {/* <AppBar position="sticky">
				<Toolbar>
					<Typography sx={{ flex: 1 }}>MUI with dnd-kit</Typography>

					<FormControlLabel
						control={
							<Switch
								checked={darkTheme}
								onChange={(_, checked) => {
									setDarkTheme(checked);
								}}
							/>
						}
						label="Dark theme"
					/>
				</Toolbar>
			</AppBar> */}
      <Box sx={{ py: 2, px: { xs: 1, sm: 2, md: 3, lg: 4, xl: 8 } }}>
        <DndContext
          autoScroll
          sensors={sensors}
          collisionDetection={closestCenter}
          onDragStart={handleDragStart}
          onDragOver={handleDragOver}
          onDragEnd={handleDragEnd}
          onDragCancel={handleDragCancel}
        >
          <SortableContext items={itemIds} strategy={() => {}}>
            <Grid columns={4} gap={2}>
              {items.map((props, index) => (
                <SortableArea
                  key={props.id}
                  {...props}
                  index={index}
                  onGridChange={(grid) => {
                    setItems((previousState) => {
                      return previousState.map((previousItem) => {
                        return previousItem.id === props.id
                          ? {
                              ...previousItem,
                              grid,
                            }
                          : previousItem;
                      });
                    });
                  }}
                >
                  <Content data={props} />
                </SortableArea>
              ))}
            </Grid>
          </SortableContext>

          <DragOverlay adjustScale={false}>
            {activeItem ? (
              <Box
                sx={{
                  display: "grid",
                  gridAutoColumns: "auto",
                  gridAutoRows: "auto",
                  height: "100%",
                }}
              >
                <Area
                  active
                  {...activeItem}
                  index={items.findIndex((item) => item.id === activeItem.id)}
                >
                  <Content data={activeItem} />
                </Area>
              </Box>
            ) : null}
          </DragOverlay>
        </DndContext>
      </Box>
    </ThemeProvider>
  );

  function handleDragStart(event) {
    setActiveId(event.active.id);
  }

  function handleDragEnd() {
    setActiveId(null);
  }

  function handleDragCancel() {
    setActiveId(null);
  }
};

export default App;
