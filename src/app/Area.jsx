"use client";

import DragHandleIcon from "@mui/icons-material/DragHandle";
import MoreVertIcon from "@mui/icons-material/MoreVert";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Card from "@mui/material/Card";
import CardHeader from "@mui/material/CardHeader";
import Divider from "@mui/material/Divider";
import IconButton from "@mui/material/IconButton";
import Menu from "@mui/material/Menu";
import Slider from "@mui/material/Slider";
import Tooltip from "@mui/material/Tooltip";
import Typography from "@mui/material/Typography";
import CardActions from "@mui/material/CardActions";
import {
  forwardRef,
  Fragment,
  memo,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";

/**
 * GridMenu - small inline menu to edit colSpan/rowSpan manually.
 */
export function GridMenu({ id, grid: initialGrid, onSave }) {
  const [anchorEl, setAnchorEl] = useState(null);
  const [grid, setGrid] = useState(initialGrid);

  useEffect(() => {
    setGrid(initialGrid);
  }, [initialGrid]);

  const open = Boolean(anchorEl);
  const handleClick = (e) => setAnchorEl(e.currentTarget);
  const handleClose = () => setAnchorEl(null);

  return (
    <Fragment>
      <Tooltip title="Grid settings">
        <IconButton
          onClick={handleClick}
          edge="end"
          sx={{ ml: 2, color: "inherit" }}
          aria-controls={open ? id : undefined}
          aria-haspopup="true"
          aria-expanded={open ? "true" : undefined}
        >
          <MoreVertIcon />
        </IconButton>
      </Tooltip>

      <Menu
        anchorEl={anchorEl}
        id={id}
        open={open}
        onClose={handleClose}
        PaperProps={{ sx: { minWidth: 244 } }}
        transformOrigin={{ horizontal: "right", vertical: "top" }}
        anchorOrigin={{ horizontal: "right", vertical: "bottom" }}
      >
        <Box px={2}>
          <Typography variant="caption">Columns</Typography>
          <Slider
            aria-label="columns"
            valueLabelDisplay="auto"
            step={1}
            min={1}
            max={12}
            value={grid.colSpan}
            onChange={(_, value) => setGrid((p) => ({ ...p, colSpan: value }))}
          />
        </Box>

        <Divider />

        <Box px={2}>
          <Typography variant="caption">Rows</Typography>
          <Slider
            aria-label="rows"
            valueLabelDisplay="auto"
            step={1}
            min={1}
            max={32}
            value={grid.rowSpan}
            onChange={(_, value) => setGrid((p) => ({ ...p, rowSpan: value }))}
          />
        </Box>

        <Divider />

        <CardActions sx={{ flexDirection: "row-reverse" }}>
          <Button
            onClick={() => {
              handleClose();
              onSave(grid);
            }}
          >
            Save
          </Button>
          <Button
            color="inherit"
            onClick={() => {
              setGrid(initialGrid);
            }}
          >
            Reset
          </Button>
        </CardActions>
      </Menu>
    </Fragment>
  );
}

/**
 * Area - a card-like item that supports resizing by dragging its bottom-right handle.
 *
 * Resize activation policy:
 *  - Pointer down installs a lightweight detector.
 *  - Detector watches pointermove but does NOT apply grid changes until:
 *      a) pointer moved beyond a small pixel threshold AND
 *      b) the pixel movement maps to a non-zero column/row delta (grid change).
 *  - When both conditions are met we:
 *      - capture the pointer (if possible),
 *      - install primary move/up handlers,
 *      - mark the interaction as a real resize and update preview immediately.
 *  - If pointerup/cancel happens before activation, we revert any preview and cleanup.
 *  - On up after activation we persist changes via onGridChange (if provided) and cleanup.
 *
 * This implementation strives to avoid the "follow and snap back" behavior by ensuring
 * activation happens only on a deliberate drag that would change grid spans.
 */
export const Area = memo(
  forwardRef(
    (
      {
        id,
        index,
        faded,
        active,
        children,
        placeholder,
        noSpan,
        bgcolor,
        dragHandle,
        progress,
        grid,
        onGridChange,
        ...props
      },
      ref,
    ) => {
      const internalRef = useRef(null);
      const [localGrid, setLocalGrid] = useState(grid);

      // keep preview synced if parent updates grid
      useEffect(() => {
        setLocalGrid(grid);
      }, [grid]);

      // forward ref to parent + internal ref
      const attachRef = useCallback(
        (node) => {
          internalRef.current = node;
          if (!ref) return;
          if (typeof ref === "function") ref(node);
          else ref.current = node;
        },
        [ref],
      );

      // Row height must match Grid.jsx gridAutoRows
      const ROW_HEIGHT = 50;

      // Breakpoints mirror Grid.jsx
      const getColumns = useCallback(() => {
        const w = window.innerWidth;
        if (w < 900) return 4;
        if (w < 1200) return 8;
        return 12;
      }, []);

      // transient state for current interaction
      // { startX, startY, startCols, startRows, parentRect, pointerId, moved, captured, lastCol, lastRow, gridSnapshot }
      const interactionRef = useRef(null);

      // calc grid target from pixel deltas
      const calcTarget = useCallback(
        (dx, dy, parentRect, startCols, startRows) => {
          const columns = getColumns();
          const parentWidth = parentRect ? parentRect.width : window.innerWidth;
          const colWidth = Math.max(1, parentWidth / columns);
          const deltaCols = Math.round(dx / colWidth);
          const deltaRows = Math.round(dy / ROW_HEIGHT);
          const targetCol = Math.max(1, startCols + deltaCols);
          const targetRow = Math.max(1, startRows + deltaRows);
          return { targetCol, targetRow };
        },
        [getColumns],
      );

      // primary move handler (applies only after activation)
      const primaryMove = useCallback(
        (ev) => {
          const s = interactionRef.current;
          if (!s) return;
          const dx = ev.clientX - s.startX;
          const dy = ev.clientY - s.startY;
          const { targetCol, targetRow } = calcTarget(
            dx,
            dy,
            s.parentRect,
            s.startCols,
            s.startRows,
          );

          // if nothing changed compared to last applied, skip
          if (s.lastCol === targetCol && s.lastRow === targetRow) return;

          s.lastCol = targetCol;
          s.lastRow = targetRow;
          s.moved = true;

          setLocalGrid({
            ...s.gridSnapshot,
            colSpan: targetCol,
            rowSpan: targetRow,
          });
        },
        [calcTarget],
      );

      // common cleanup used on cancel/up/unmount
      const finishInteraction = useCallback(
        (event) => {
          const s = interactionRef.current;
          const el = internalRef.current;

          if (
            s &&
            s.captured &&
            el &&
            typeof el.releasePointerCapture === "function"
          ) {
            try {
              const pid = s.pointerId ?? (event && event.pointerId);
              if (pid != null) el.releasePointerCapture(pid);
            } catch (err) {
              // ignore release errors
            }
          }

          // remove listeners
          window.removeEventListener("pointermove", primaryMove);
          window.removeEventListener("pointerup", finishInteraction);
          window.removeEventListener("pointercancel", finishInteraction);
          window.removeEventListener("mouseup", finishInteraction);
          window.removeEventListener("blur", finishInteraction);
          document.removeEventListener("visibilitychange", finishInteraction);

          if (s) {
            if (s.moved) {
              // persist if we actually resized
              if (typeof onGridChange === "function") {
                onGridChange({ colSpan: s.lastCol, rowSpan: s.lastRow });
              }
            } else {
              // revert any preview
              setLocalGrid(s.gridSnapshot);
            }
          }

          interactionRef.current = null;
        },
        [primaryMove, onGridChange],
      );

      // pointerdown on the handle: install a detector that only activates the primary handlers
      const handlePointerDown = useCallback(
        (ev) => {
          // ignore non-primary buttons (still allow touch which often has button==0/undefined)
          if (ev.button && ev.button !== 0) return;
          ev.stopPropagation();

          const el = internalRef.current;
          if (!el) return;

          const parentRect = el.parentElement
            ? el.parentElement.getBoundingClientRect()
            : null;

          // snapshot current grid
          const snapshot = { ...localGrid };

          // initialize interaction state
          const s = {
            startX: ev.clientX,
            startY: ev.clientY,
            startCols: snapshot.colSpan ?? 1,
            startRows: snapshot.rowSpan ?? 1,
            parentRect,
            pointerId: ev.pointerId,
            moved: false,
            captured: false,
            lastCol: snapshot.colSpan ?? 1,
            lastRow: snapshot.rowSpan ?? 1,
            gridSnapshot: snapshot,
          };
          interactionRef.current = s;

          // detector: wait for either a small pixel threshold AND a grid-cell change
          const PIXEL_THRESHOLD = 6;

          const detectMove = (moveEvent) => {
            const cur = interactionRef.current;
            if (!cur) return;
            const dx = moveEvent.clientX - cur.startX;
            const dy = moveEvent.clientY - cur.startY;

            // require minimal movement
            if (Math.hypot(dx, dy) < PIXEL_THRESHOLD) return;

            const { targetCol, targetRow } = calcTarget(
              dx,
              dy,
              cur.parentRect,
              cur.startCols,
              cur.startRows,
            );

            // only activate when a grid-cell change will occur
            if (targetCol === cur.startCols && targetRow === cur.startRows)
              return;

            // activate primary mode: try to capture pointer and install primary handlers
            try {
              const pid = cur.pointerId ?? moveEvent.pointerId;
              el.setPointerCapture(pid);
              cur.captured = true;
            } catch (err) {
              cur.captured = false;
            }

            // record that we've activated and apply first preview
            cur.moved = true;
            cur.lastCol = targetCol;
            cur.lastRow = targetRow;
            setLocalGrid({
              ...cur.gridSnapshot,
              colSpan: targetCol,
              rowSpan: targetRow,
            });

            // switch listeners
            window.removeEventListener("pointermove", detectMove);
            window.removeEventListener("pointerup", cancelDetect);
            window.removeEventListener("pointercancel", cancelDetect);

            window.addEventListener("pointermove", primaryMove);
            window.addEventListener("pointerup", finishInteraction);
            window.addEventListener("pointercancel", finishInteraction);
            window.addEventListener("mouseup", finishInteraction);
            window.addEventListener("blur", finishInteraction);
            document.addEventListener("visibilitychange", finishInteraction);
          };

          const cancelDetect = (upEvent) => {
            // user released before actual activation -> revert and cleanup
            window.removeEventListener("pointermove", detectMove);
            window.removeEventListener("pointerup", cancelDetect);
            window.removeEventListener("pointercancel", cancelDetect);

            // revert preview immediately
            const cur = interactionRef.current;
            if (cur) setLocalGrid(cur.gridSnapshot);

            // ensure no primary handlers left
            window.removeEventListener("pointermove", primaryMove);
            window.removeEventListener("pointerup", finishInteraction);
            window.removeEventListener("pointercancel", finishInteraction);

            // release capture if set
            try {
              if (
                cur?.captured &&
                el &&
                typeof el.releasePointerCapture === "function"
              ) {
                const pid = cur.pointerId ?? (upEvent && upEvent.pointerId);
                if (pid != null) el.releasePointerCapture(pid);
              }
            } catch (err) {
              // ignore
            }

            interactionRef.current = null;
          };

          // store detector handlers in case we need to remove them by reference
          s.detectMove = detectMove;
          s.cancelDetect = cancelDetect;

          // attach detector listeners (lightweight)
          window.addEventListener("pointermove", detectMove);
          window.addEventListener("pointerup", cancelDetect);
          window.addEventListener("pointercancel", cancelDetect);
        },
        [localGrid, calcTarget, primaryMove, finishInteraction],
      );

      // cleanup on unmount to avoid lingering listeners / capture
      useEffect(() => {
        return () => {
          try {
            const s = interactionRef.current;
            const el = internalRef.current;
            if (
              s?.captured &&
              el &&
              typeof el.releasePointerCapture === "function"
            ) {
              el.releasePointerCapture(s.pointerId);
            }
          } catch (err) {
            // ignore
          }
          window.removeEventListener("pointermove", primaryMove);
          window.removeEventListener("pointerup", finishInteraction);
          window.removeEventListener("pointercancel", finishInteraction);
          window.removeEventListener("mouseup", finishInteraction);
          window.removeEventListener("blur", finishInteraction);
          document.removeEventListener("visibilitychange", finishInteraction);

          // also remove detector handlers if they are still present
          const s = interactionRef.current;
          if (s?.detectMove)
            window.removeEventListener("pointermove", s.detectMove);
          if (s?.cancelDetect) {
            window.removeEventListener("pointerup", s.cancelDetect);
            window.removeEventListener("pointercancel", s.cancelDetect);
          }
          interactionRef.current = null;
        };
      }, [primaryMove, finishInteraction]);

      return (
        <Card
          ref={attachRef}
          elevation={active ? 6 : 0}
          variant={active ? "elevation" : "outlined"}
          {...props}
          sx={(theme) => ({
            pointerEvents: active ? "none" : undefined,
            userSelect: "none",
            WebkitUserSelect: "none",
            MozUserSelect: "none",
            msUserSelect: "none",
            position: "relative",
            display: "flex",
            flex: 1,
            flexDirection: "column",
            overflow: "hidden",
            opacity: faded ? 0.8 : 1,
            transformOrigin: "0 0",
            bgcolor: placeholder ? "action.hover" : bgcolor,
            color: bgcolor ? theme.palette.getContrastText(bgcolor) : undefined,
            transform:
              "translate3d(var(--translate-x, 0), var(--translate-y, 0), 0)",
            transition: "var(--transition, none)",
            gridRow: noSpan
              ? undefined
              : `span ${localGrid?.rowSpan ?? grid.rowSpan}`,
            gridColumn: {
              xs: noSpan
                ? undefined
                : `span ${Math.min(4, localGrid?.colSpan ?? grid.colSpan)}`,
              sm: noSpan
                ? undefined
                : `span ${Math.min(4, localGrid?.colSpan ?? grid.colSpan)}`,
              md: noSpan
                ? undefined
                : `span ${Math.min(8, localGrid?.colSpan ?? grid.colSpan)}`,
              lg: noSpan
                ? undefined
                : `span ${localGrid?.colSpan ?? grid.colSpan}`,
            },
            // ensure images inside the card are not draggable/selectable during interactions
            "& img": {
              userSelect: "none",
              WebkitUserSelect: "none",
              MozUserSelect: "none",
              msUserSelect: "none",
              pointerEvents: "none",
              WebkitUserDrag: "none",
            },
          })}
        >
          {placeholder ? null : (
            <>
              <CardHeader
                avatar={
                  <IconButton
                    {...dragHandle}
                    edge="start"
                    sx={{ color: "inherit" }}
                  >
                    <DragHandleIcon />
                  </IconButton>
                }
                action={
                  <GridMenu
                    id={`grid-menu-${id}`}
                    grid={localGrid ?? grid}
                    onSave={onGridChange}
                  />
                }
                sx={{ py: 0, ".MuiCardHeader-action": { m: 0 } }}
              />
              {children}

              {/* bottom-right resize handle */}
              <Box
                component="div"
                onPointerDown={handlePointerDown}
                sx={{
                  position: "absolute",
                  right: 8,
                  bottom: 8,
                  width: 20,
                  height: 20,
                  bgcolor: "rgba(0,0,0,0.12)",
                  border: "1px solid rgba(0,0,0,0.2)",
                  borderRadius: 0.5,
                  cursor: "se-resize",
                  zIndex: 10,
                }}
              />
            </>
          )}
        </Card>
      );
    },
  ),
);
