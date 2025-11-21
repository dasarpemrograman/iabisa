// src/components/chatbot-panel.tsx
"use client";

import React, { useState, useRef, useEffect } from "react";
import {
  Send,
  Maximize2,
  Minimize2,
  PlusCircle,
  Activity,
  Database,
  Map as MapIcon,
} from "lucide-react";
import { useDashboard } from "@/context/dashboard-context"; // Import the hook
import DynamicWidget from "./dynamic-chart"; // Import the renderer

// --- Types ---
interface Message {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  type?: "text" | "chart" | "map" | "status" | "error";
  data?: any; // For chart/map data
  componentName?: string; // To identify chart type
  artifacts?: { title: string; content: string; type: "sql" | "json" }[];
  timestamp: Date;
}

// --- Main Component ---
export default function ChatbotPanel({
  fullscreen = false,
  onFullscreen,
  onExitFullscreen,
}: {
  fullscreen?: boolean;
  onFullscreen?: () => void;
  onExitFullscreen?: () => void;
}) {
  const { addItem } = useDashboard(); // Connect to Dashboard
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      role: "assistant",
      content:
        "Hello! I am your Agentic BI Assistant. Ask me about your data, or ask to generate charts and maps.",
      type: "text",
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [currentStatus, setCurrentStatus] = useState<string>("");

  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, currentStatus]);

  // --- Handler: Send Message ---
  const handleSendMessage = async () => {
    if (!input.trim() || isStreaming) return;

    const userMsg: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input,
      type: "text",
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setIsStreaming(true);
    setCurrentStatus("🧠 Starting agent...");

    // Prepare history for backend
    const history = messages.map((m) => ({
      role: m.role === "user" ? "user" : "assistant",
      content: m.content,
    }));

    try {
      const response = await fetch(
        "http://localhost:8000/generate-chart-stream",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: userMsg.content, history }),
        },
      );

      if (!response.body) throw new Error("No response body");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      // Create a placeholder message for the assistant's response
      const responseId = (Date.now() + 1).toString();
      const assistantMsg: Message = {
        id: responseId,
        role: "assistant",
        content: "",
        type: "text",
        timestamp: new Date(),
        artifacts: [],
      };

      setMessages((prev) => [...prev, assistantMsg]);

      // Stream Loop
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n"); // SSE messages are separated by double newline
        buffer = lines.pop() || ""; // Keep partial line

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const jsonStr = line.replace("data: ", "");
            try {
              const event = JSON.parse(jsonStr);
              handleStreamEvent(event, responseId);
            } catch (e) {
              console.error("Error parsing stream event", e);
            }
          }
        }
      }
    } catch (error) {
      console.error(error);
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now().toString(),
          role: "assistant",
          content:
            "❌ Connection failed. Please check if the backend is running.",
          type: "error",
          timestamp: new Date(),
        },
      ]);
    } finally {
      setIsStreaming(false);
      setCurrentStatus("");
    }
  };

  // --- Handler: Process Stream Events ---
  const handleStreamEvent = (event: any, msgId: string) => {
    const { type, label, state, content, view } = event;

    if (type === "status") {
      setCurrentStatus(state === "running" ? label : "");
    } else if (type === "log") {
      // Optional logging
    } else if (type === "artifact") {
      setMessages((prev) =>
        prev.map((m) => {
          if (m.id === msgId) {
            return {
              ...m,
              artifacts: [
                ...(m.artifacts || []),
                { title: label, content, type: view },
              ],
            };
          }
          return m;
        }),
      );
    } else if (type === "final") {
      // Finalize the message
      setMessages((prev) =>
        prev.map((m) => {
          if (m.id === msgId) {
            // If content is complex (chart/map), store data
            if (view === "chart" || view === "map") {
              return {
                ...m,
                type: view,
                content: content.component_name || "Visualization",
                // FIX STARTS HERE: Pass full content for maps, just rows for charts
                data: view === "map" ? content : content.data,
                // FIX ENDS HERE
                componentName: content.component_name,
              };
            }
            // Else it's text
            return { ...m, content: content, type: "text" };
          }
          return m;
        }),
      );
    }
  };

  // --- Helper: Add to Dashboard ---
  const handleAddToDashboard = (msg: Message) => {
    addItem({
      widget: "dynamic-chart",
      bgcolor: "#ffffff",
      grid: { colSpan: 4, rowSpan: 8 },
      data: msg.data,
      title: msg.componentName || "New Insight",
      type: msg.type, // <--- Ensure this is present
    });
  };

  return (
    <div className="flex h-full flex-col border-l border-gray-200 bg-white">
      {/* Header */}
      <div className="flex h-14 flex-shrink-0 items-center justify-between border-b border-gray-200 bg-gray-50 px-4">
        <div>
          <h2 className="flex items-center gap-2 font-semibold text-gray-900">
            <Activity size={18} className="text-blue-600" /> Agentic BI
          </h2>
        </div>
        <button
          onClick={fullscreen ? onExitFullscreen : onFullscreen}
          className="rounded-lg p-1.5 hover:bg-gray-200"
        >
          {fullscreen ? <Minimize2 size={18} /> : <Maximize2 size={18} />}
        </button>
      </div>

      {/* Messages */}
      <div className="flex-1 space-y-4 overflow-y-auto bg-gray-50/50 p-4">
        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`flex flex-col ${msg.role === "user" ? "items-end" : "items-start"}`}
          >
            {/* Message Bubble */}
            <div
              className={`max-w-[90%] rounded-2xl px-4 py-3 shadow-sm ${
                msg.role === "user"
                  ? "rounded-br-none bg-blue-600 text-white"
                  : "rounded-bl-none border border-gray-200 bg-white text-gray-800"
              }`}
            >
              {/* Text Content */}
              <div className="prose prose-sm max-w-none">
                {msg.type === "text" || msg.type === "error"
                  ? msg.content
                  : null}
              </div>

              {/* Visualization Content */}
              {(msg.type === "chart" || msg.type === "map") && (
                <div className="mt-2 w-full min-w-[300px]">
                  <div className="mb-2 flex items-center justify-between">
                    <span className="text-xs font-semibold text-gray-500 uppercase">
                      {msg.type}
                    </span>
                    <button
                      onClick={() => handleAddToDashboard(msg)}
                      className="flex items-center gap-1 rounded bg-blue-50 px-2 py-1 text-xs text-blue-600 transition-colors hover:bg-blue-100"
                    >
                      <PlusCircle size={12} /> Add to Dashboard
                    </button>
                  </div>
                  <div className="h-64 rounded-lg border border-gray-100 bg-white p-2">
                    <DynamicWidget
                      data={msg.data}
                      type={
                        msg.type === "map"
                          ? "map"
                          : msg.componentName?.toLowerCase().includes("bar")
                            ? "barchart"
                            : "linechart"
                      }
                      title={msg.componentName}
                    />
                  </div>
                </div>
              )}

              {/* Artifacts (SQL/JSON) */}
              {msg.artifacts && msg.artifacts.length > 0 && (
                <div className="mt-3 space-y-2">
                  {msg.artifacts.map((art, idx) => (
                    <details key={idx} className="group">
                      <summary className="flex cursor-pointer items-center gap-1 text-xs text-gray-500 hover:text-blue-600">
                        <Database size={10} /> {art.title}
                      </summary>
                      <pre className="mt-2 overflow-x-auto rounded bg-gray-900 p-2 text-[10px] text-green-400">
                        {art.content}
                      </pre>
                    </details>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}

        {/* Status Indicator */}
        {isStreaming && (
          <div className="flex animate-pulse items-center gap-2 px-4 text-xs text-gray-500">
            <div className="h-2 w-2 animate-bounce rounded-full bg-blue-500" />
            {currentStatus}
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="flex-shrink-0 border-t border-gray-200 bg-white p-4">
        <div className="relative">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) =>
              e.key === "Enter" && !isStreaming && handleSendMessage()
            }
            placeholder="Ask a question (e.g., 'Show patient trends 2024')..."
            className="w-full rounded-xl border border-gray-300 py-3 pr-12 pl-4 text-sm shadow-sm outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
            disabled={isStreaming}
          />
          <button
            onClick={handleSendMessage}
            disabled={!input.trim() || isStreaming}
            className="absolute top-2 right-2 rounded-lg bg-blue-600 p-1.5 text-white transition-all hover:bg-blue-700 disabled:opacity-50"
          >
            <Send size={16} />
          </button>
        </div>
      </div>
    </div>
  );
}
