import { useSortable } from "@dnd-kit/sortable";
import { X } from "lucide-react";
import { useDashboard } from "@/context/dashboard-context";

import { Area } from "./Area";

export const SortableArea = ({ id, children, className, ...props }) => {
  const {
    attributes,
    isDragging,
    listeners,
    setNodeRef,
    transform,
    transition,
  } = useSortable({ id });

  const { removeItem } = useDashboard();

  return (
    <Area
      ref={setNodeRef}
      id={id}
      {...props}
      // Add 'group' here to enable hover detection for children
      className={`group relative ${className || ""}`}
      style={{
        "--translate-x": transform ? `${transform.x}px` : 0,
        "--translate-y": transform ? `${transform.y}px` : 0,
        "--transition": transition ?? "none",
      }}
      placeholder={isDragging}
      dragHandle={{ ...attributes, ...listeners }}
    >
      {children}

      {/* Minimalist Remove Button */}
      <button
        className="absolute top-3 right-3 z-50 text-gray-300 opacity-0 transition-opacity group-hover:opacity-100 hover:text-red-500"
        onPointerDown={(e) => e.stopPropagation()}
        onClick={(e) => {
          e.stopPropagation();
          removeItem(id);
        }}
        title="Remove widget"
      >
        <X size={18} />
      </button>
    </Area>
  );
};
