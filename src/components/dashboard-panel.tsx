"use client";

import { Maximize2, Minimize2 } from "lucide-react";
import DashBoardContent from "./dashboard-content";

interface DashboardPanelProps {
  fullscreen?: boolean;
  onFullscreen?: () => void;
  onExitFullscreen?: () => void;
}

export default function DashboardPanel({
  fullscreen = false,
  onFullscreen,
  onExitFullscreen,
}: DashboardPanelProps) {
  return (
    <div className="flex h-full flex-col border-r border-gray-200 bg-white">
      {/* Content Area */}
      <div className="overflow-auto">
        <div className="h-full items-center">
          <DashBoardContent />
        </div>
      </div>
    </div>
  );
}
