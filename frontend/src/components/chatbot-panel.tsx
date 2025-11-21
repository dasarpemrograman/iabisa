"use client";

import React, { useState, useRef, useEffect } from "react";
import {
  Send,
  Bot,
  User,
  Sparkles,
  Loader2,
  BarChart2,
  AlertCircle,
  PlusCircle,
} from "lucide-react";
import { useDashboard } from "@/context/dashboard-context";
import ReactMarkdown from "react-markdown";
import DynamicWidget from "./dynamic-chart";

interface Message {
  role: "user" | "assistant" | "system";
  content: string | any;
  type?: "text" | "status" | "error" | "chart" | "map";
  steps?: {
    label: string;
    status: "running" | "complete" | "error";
    content?: string;
  }[];
  title?: string;
}

export default function ChatbotPanel() {
  const { addItem } = useDashboard();
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "system",
      content:
        "Hello! I am your BI Assistant. Ask me to predict facility growth or query the database.",
      type: "text",
    },
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage = input;
    setInput("");
    setMessages((prev) => [
      ...prev,
      { role: "user", content: userMessage, type: "text" },
    ]);
    setIsLoading(true);

    const history = messages
      .filter((m) => m.role !== "system")
      .map((m) => ({
        role: m.role,
        content:
          typeof m.content === "string" ? m.content : JSON.stringify(m.content),
      }));

    try {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/generate-chart-stream`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: userMessage, history }),
        },
      );

      if (!response.body) throw new Error("No response body");

      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "", steps: [], type: "text" },
      ]);

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const jsonStr = line.replace("data: ", "").trim();
            if (!jsonStr) continue;

            try {
              const event = JSON.parse(jsonStr);
              processStreamEvent(event);
            } catch (e) {
              console.error("Error parsing stream event:", e);
            }
          }
        }
      }
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        { role: "system", content: `Error: ${error}`, type: "error" },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const processStreamEvent = (event: any) => {
    setMessages((prev) => {
      const newMsgs = [...prev];
      const lastMsg = newMsgs[newMsgs.length - 1];

      // 1. Handle Intermediate Steps (Status updates)
      if (["status", "log", "artifact"].includes(event.type)) {
        const steps = lastMsg.steps || [];
        const existingStepIndex = steps.findIndex(
          (s) => s.label === event.label,
        );

        const newStep = {
          label: event.label,
          status: event.state || "running",
          content: event.content,
        };

        if (existingStepIndex >= 0) steps[existingStepIndex] = newStep;
        else steps.push(newStep);

        lastMsg.steps = steps;
      }

      // 2. Handle Final Response
      if (event.type === "final") {
        // Determine Type
        const isChart = event.view === "chart" || event.view === "map";
        lastMsg.type = isChart ? event.view : "text";
        lastMsg.content = event.content;
        lastMsg.title = isChart
          ? event.content.title || "AI Insight"
          : undefined;

        // --- FIX: FORCE ALL STEPS TO COMPLETE ---
        if (lastMsg.steps) {
          lastMsg.steps = lastMsg.steps.map((step) => ({
            ...step,
            status: step.status === "running" ? "complete" : step.status,
          }));
        }
      }
      return newMsgs;
    });
  };

  const handleManualAdd = (msg: Message) => {
    const widgetType = msg.type === "map" ? "map" : "dynamic-chart";
    addItem(widgetType, msg.title || "New Widget", msg.content);
  };

  return (
    <div className="z-50 flex h-full w-full max-w-md flex-col border-l border-gray-200 bg-white font-sans text-gray-800 shadow-xl">
      {/* Header */}
      <div className="sticky top-0 z-10 flex items-center gap-2 border-b border-gray-100 bg-white/80 p-4 backdrop-blur-sm">
        <div className="rounded-lg bg-blue-100 p-1.5">
          <Bot className="h-5 w-5 text-blue-600" />
        </div>
        <div>
          <h2 className="text-sm font-bold text-gray-900">Agentic BI</h2>
          <p className="text-[10px] text-gray-500">AI-Powered Analytics</p>
        </div>
      </div>

      {/* Messages Area */}
      <div className="scrollbar-thin scrollbar-thumb-gray-300 flex-1 space-y-6 overflow-y-auto p-4">
        {messages.map((msg, idx) => (
          <div
            key={idx}
            className={`flex gap-3 ${msg.role === "user" ? "flex-row-reverse" : ""}`}
          >
            {/* Avatar */}
            <div
              className={`flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full shadow-sm ${
                msg.role === "user"
                  ? "bg-blue-600 text-white"
                  : msg.role === "system"
                    ? "bg-red-50 text-red-600"
                    : "border border-gray-200 bg-white text-indigo-600"
              }`}
            >
              {msg.role === "user" ? (
                <User size={14} />
              ) : msg.role === "system" ? (
                <AlertCircle size={14} />
              ) : (
                <Sparkles size={14} />
              )}
            </div>

            {/* Bubble */}
            <div
              className={`flex max-w-[90%] flex-col space-y-1 ${msg.role === "user" ? "items-end" : "items-start"}`}
            >
              {/* Workflow Steps */}
              {msg.steps && msg.steps.length > 0 && (
                <div className="mb-2 w-full space-y-1">
                  {msg.steps.map((step, sIdx) => (
                    <div
                      key={sIdx}
                      className="flex items-center gap-2 rounded border border-gray-100 bg-gray-50 px-2 py-1.5 text-[10px] text-gray-500"
                    >
                      {step.status === "running" && (
                        <Loader2 className="h-3 w-3 animate-spin text-blue-500" />
                      )}
                      {step.status === "complete" && (
                        <div className="h-2 w-2 rounded-full bg-green-500" />
                      )}
                      {step.status === "error" && (
                        <div className="h-2 w-2 rounded-full bg-red-500" />
                      )}
                      <span className="truncate font-medium">{step.label}</span>
                    </div>
                  ))}
                </div>
              )}

              {/* Main Content */}
              <div
                className={`w-full rounded-2xl px-4 py-3 text-sm leading-relaxed shadow-sm ${
                  msg.role === "user"
                    ? "rounded-br-none bg-blue-600 text-white"
                    : msg.type === "error"
                      ? "border border-red-100 bg-red-50 text-red-800"
                      : "rounded-bl-none border border-gray-200 bg-gray-100 text-gray-800"
                }`}
              >
                {msg.type === "chart" || msg.type === "map" ? (
                  <div className="flex w-full min-w-[280px] flex-col gap-3">
                    <div className="mb-1 flex items-center justify-between border-b border-gray-200 pb-2">
                      <div className="flex items-center gap-2 font-semibold text-gray-700">
                        <BarChart2 className="h-4 w-4 text-blue-600" />
                        <span className="text-xs">
                          {msg.title || "Generated Chart"}
                        </span>
                      </div>
                      <button
                        onClick={() => handleManualAdd(msg)}
                        className="flex items-center gap-1 rounded border border-gray-300 bg-white px-2 py-1 text-[10px] shadow-sm transition-all hover:border-blue-200 hover:bg-blue-50 hover:text-blue-600"
                      >
                        <PlusCircle size={12} />
                        Add to Dashboard
                      </button>
                    </div>
                    <div className="h-48 w-full rounded border border-gray-200 bg-white p-2">
                      <DynamicWidget data={msg.content} />
                    </div>
                  </div>
                ) : (
                  <div className="prose prose-stone prose-sm max-w-none">
                    <ReactMarkdown>
                      {typeof msg.content === "string" ? msg.content : ""}
                    </ReactMarkdown>
                  </div>
                )}
              </div>
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="border-t border-gray-100 bg-white p-4">
        <div className="relative flex items-center rounded-xl border border-gray-200 bg-gray-50 shadow-inner transition-all focus-within:border-blue-500 focus-within:ring-1 focus-within:ring-blue-500">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSend()}
            placeholder="Ask for a prediction..."
            className="flex-1 border-none bg-transparent px-4 py-3 text-sm text-gray-800 placeholder-gray-400 focus:ring-0"
            disabled={isLoading}
          />
          <button
            onClick={handleSend}
            disabled={isLoading || !input.trim()}
            className={`mr-2 rounded-lg p-2 transition-all ${
              input.trim() && !isLoading
                ? "bg-blue-600 text-white shadow-md hover:bg-blue-700"
                : "cursor-not-allowed text-gray-400"
            }`}
          >
            {isLoading ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Send className="h-4 w-4" />
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
