"use client";

import type React from "react";

import { useState, useRef } from "react";
import DashboardPanel from "./dashboard-panel";
import ChatbotPanel from "./chatbot-panel";
import DashboardContent from "./dashboard-content";

export default function DashboardLayout() {
  const [dividerPos, setDividerPos] = useState(50);
  const [isDragging, setIsDragging] = useState(false);
  const [fullscreen, setFullscreen] = useState<"dashboard" | "chatbot" | null>(
    null,
  );
  const containerRef = useRef<HTMLDivElement>(null);

  const handleMouseDown = () => {
    setIsDragging(true);
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleMouseMove = (e: React.MouseEvent) => {
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
      <div
        className="flex-1 overflow-hidden"
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      >
        <div className="relative h-full w-full">
          <DashboardPanel
            fullscreen
            onExitFullscreen={() => setFullscreen(null)}
          />
        </div>
      </div>
    );
  }

  if (fullscreen === "chatbot") {
    return (
      <div
        className="flex-1 overflow-hidden"
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      >
        <div className="relative h-full w-full">
          <ChatbotPanel
            fullscreen
            onExitFullscreen={() => setFullscreen(null)}
          />
        </div>
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className="flex flex-1 overflow-hidden"
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
    >
      {/* Dashboard Panel */}
      <div
        style={{ width: `${dividerPos}%` }}
        className="relative overflow-hidden"
      >
        <DashboardPanel onFullscreen={() => setFullscreen("dashboard")} />
      </div>

      {/* Divider */}
      <div
        onMouseDown={handleMouseDown}
        className={`w-1 cursor-col-resize bg-linear-to-b from-transparent via-gray-300 to-transparent transition-colors hover:bg-linear-to-b hover:from-transparent hover:via-gray-400 hover:to-transparent ${
          isDragging
            ? "bg-linear-to-b from-transparent via-blue-400 to-transparent"
            : ""
        }`}
        style={{
          background: isDragging
            ? "linear-gradient(to bottom, transparent, #2A4491, transparent)"
            : "linear-gradient(to bottom, transparent, #d1d5db, transparent)",
        }}
      />

      {/* Chatbot Panel */}
      <div
        style={{ width: `${100 - dividerPos}%` }}
        className="relative overflow-hidden"
      >
        <ChatbotPanel onFullscreen={() => setFullscreen("chatbot")} />
      </div>
    </div>
  );
}
