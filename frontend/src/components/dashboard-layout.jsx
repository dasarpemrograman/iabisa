"use client";

import { useState, useRef } from "react";
import DashboardPanel from "./dashboard-panel";
import ChatbotPanel from "./chatbot-panel";

export default function DashboardLayout() {
  // HAPUS semua generic type <...> biar tidak crash di .jsx
  const [dividerPos, setDividerPos] = useState(60); 
  const [isDragging, setIsDragging] = useState(false);
  const [fullscreen, setFullscreen] = useState(null);
  const containerRef = useRef(null); // Sudah bersih dari <HTMLDivElement>

  const handleMouseDown = () => {
    setIsDragging(true);
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleMouseMove = (e) => {
    if (!isDragging || !containerRef.current) return;

    const container = containerRef.current;
    const rect = container.getBoundingClientRect();
    const newPos = ((e.clientX - rect.left) / rect.width) * 100;

    if (newPos > 20 && newPos < 80) {
      setDividerPos(newPos);
    }
  };

  if (fullscreen === "dashboard") {
    return (
      <div className="flex-1 overflow-hidden bg-gray-50">
        <DashboardPanel
          fullscreen
          onExitFullscreen={() => setFullscreen(null)}
        />
      </div>
    );
  }

  if (fullscreen === "chatbot") {
    return (
      <div className="flex-1 overflow-hidden bg-gray-50">
        <ChatbotPanel
          fullscreen
          onExitFullscreen={() => setFullscreen(null)}
        />
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className="flex flex-1 overflow-hidden bg-gradient-to-br from-gray-50 via-emerald-50/30 to-gray-100"
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
    >
      {/* Dashboard Panel (Left) */}
      <div
        style={{ width: `${dividerPos}%` }}
        className="relative overflow-hidden shadow-[2px_0_20px_rgba(0,0,0,0.05)] z-10"
      >
        <DashboardPanel onFullscreen={() => setFullscreen("dashboard")} />
      </div>

      {/* Divider (Draggable) */}
      <div
        onMouseDown={handleMouseDown}
        className={`relative z-20 -ml-0.5 w-1.5 cursor-col-resize transition-all duration-300 flex items-center justify-center group ${
            isDragging ? "scale-x-125" : "hover:scale-x-110"
        }`}
      >
        <div 
            className={`h-full w-0.5 rounded-full transition-all duration-300 ${
                isDragging 
                ? "bg-emerald-500 shadow-[0_0_10px_#10B981]" 
                : "bg-gray-300 group-hover:bg-emerald-300"
            }`}
        />
      </div>

      {/* Chatbot Panel (Right) */}
      <div
        style={{ width: `${100 - dividerPos}%` }}
        className="relative overflow-hidden"
      >
        <ChatbotPanel onFullscreen={() => setFullscreen("chatbot")} />
      </div>
    </div>
  );
}