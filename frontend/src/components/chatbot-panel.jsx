"use client";

import { useState, useRef, useEffect } from "react";
import {
  Send,
  Bot,
  User,
  Sparkles,
  Loader2,
  BarChart2,
  AlertCircle,
  PlusCircle,
  Map as MapIcon,
} from "lucide-react";
import { useDashboard } from "@/context/dashboard-context";
import ReactMarkdown from "react-markdown";
import DynamicWidget from "./dynamic-chart";
import BPJSIndonesiaMap from "./map";

export default function ChatbotPanel({ fullscreen, onExitFullscreen }) {
  const { addItem } = useDashboard();
  
  // GANTI useState input dengan useRef sesuai permintaan
  const inputRef = useRef(null);
  
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content: "Halo! Saya adalah asisten chatbot serbabisa IABISA. Saya bisa menjawab, membuat visualisasi, atau memprediksi apapun yang anda mau!",
      type: "text",
    },
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = async () => {
    // Ambil value langsung dari ref
    const userMessage = inputRef.current?.value;

    if (!userMessage || !userMessage.trim() || isLoading) return;

    // Kosongkan input via ref (tanpa re-render)
    if (inputRef.current) inputRef.current.value = "";

    setMessages((prev) => [
      ...prev,
      { role: "user", content: userMessage, type: "text" },
    ]);
    setIsLoading(true);

    const history = messages
      .filter((m) => m.role !== "system")
      .map((m) => ({
        role: m.role,
        content: typeof m.content === "string" ? m.content : JSON.stringify(m.content),
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

  const processStreamEvent = (event) => {
    setMessages((prev) => {
      const newMsgs = [...prev];
      const lastMsg = newMsgs[newMsgs.length - 1];

      if (["status", "log", "artifact"].includes(event.type)) {
        const steps = lastMsg.steps || [];
        const existingStepIndex = steps.findIndex((s) => s.label === event.label);

        const newStep = {
          label: event.label,
          status: event.state || "running",
          content: event.content,
        };

        if (existingStepIndex >= 0) steps[existingStepIndex] = newStep;
        else steps.push(newStep);

        lastMsg.steps = steps;
      }

      if (event.type === "final") {
        const isChart = event.view === "chart" || event.view === "map";
        lastMsg.type = isChart ? event.view : "text";
        lastMsg.content = event.content;
        lastMsg.title = isChart ? event.content.title || "AI Insight" : undefined;

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

  const handleManualAdd = (msg) => {
    const widgetType = msg.type === "map" ? "map" : "dynamic-chart";
    addItem(widgetType, msg.title || "New Widget", msg.content);
  };

  return (
    <div className="z-50 flex h-full w-full flex-col border-l border-white/20 bg-white/90 font-sans text-gray-800 shadow-2xl backdrop-blur-xl">


      {/* Messages Area */}
      <div className="flex-1 space-y-6 overflow-y-auto p-4 scrollbar-thin scrollbar-thumb-emerald-200 scrollbar-track-transparent">
        {messages.map((msg, idx) => (
          <div key={idx} className={`flex gap-3 ${msg.role === "user" ? "flex-row-reverse" : ""}`}>
            <div
              className={`flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-full shadow-md transition-transform duration-300 hover:scale-105 ${
                msg.role === "user"
                  ? "bg-gradient-to-br from-emerald-500 to-teal-600 text-white shadow-emerald-500/30"
                  : msg.role === "system"
                  ? "bg-rose-50 text-rose-500"
                  : "border border-emerald-100 bg-white text-emerald-600"
              }`}
            >
              {msg.role === "user" ? <User size={16} /> : msg.role === "system" ? <AlertCircle size={16} /> : <Sparkles size={16} />}
            </div>

            <div className={`flex max-w-[90%] flex-col space-y-1 ${msg.role === "user" ? "items-end" : "items-start"}`}>
              {msg.steps && msg.steps.length > 0 && (
                <div className="mb-2 w-full space-y-1.5">
                  {msg.steps.map((step, sIdx) => (
                    <div key={sIdx} className="flex items-center gap-2 rounded-lg border border-emerald-50 bg-emerald-50/50 px-3 py-1.5 text-[10px] text-emerald-700 backdrop-blur-sm">
                      {step.status === "running" && <Loader2 className="h-3 w-3 animate-spin text-emerald-500" />}
                      {step.status === "complete" && <div className="h-2 w-2 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.6)]" />}
                      {step.status === "error" && <div className="h-2 w-2 rounded-full bg-rose-500" />}
                      <span className="truncate font-semibold tracking-wide">{step.label}</span>
                    </div>
                  ))}
                </div>
              )}

              <div
                className={`w-full rounded-2xl px-5 py-3.5 text-sm leading-relaxed shadow-sm ${
                  msg.role === "user"
                    ? "rounded-br-none bg-gradient-to-br from-emerald-500 to-teal-600 text-white shadow-emerald-500/25"
                    : msg.type === "error"
                    ? "border border-rose-100 bg-rose-50 text-rose-800"
                    : "rounded-bl-none border border-gray-100 bg-white text-gray-700 shadow-gray-100"
                }`}
              >
                {msg.type === "chart" || msg.type === "map" ? (
                  <div className="flex w-full min-w-[280px] flex-col gap-3">
                    <div className="mb-1 flex items-center justify-between border-b border-gray-100 pb-2">
                      <div className="flex items-center gap-2 font-bold text-emerald-800">
                        {msg.type === "map" ? <MapIcon className="h-4 w-4 text-emerald-600" /> : <BarChart2 className="h-4 w-4 text-emerald-600" />}
                        <span className="text-xs">{msg.title || "Generated Insight"}</span>
                      </div>
                      <button
                        onClick={() => handleManualAdd(msg)}
                        className="flex items-center gap-1 rounded-md border border-emerald-200 bg-emerald-50 px-2 py-1 text-[10px] font-medium text-emerald-700 transition-all hover:bg-emerald-100 hover:shadow-sm"
                      >
                        <PlusCircle size={12} />
                        Add to Dashboard
                      </button>
                    </div>
                    <div className="relative h-48 w-full overflow-hidden rounded-lg border border-gray-100 bg-white p-2 shadow-inner">
                      {msg.type === "map" ? <BPJSIndonesiaMap data={msg.content} /> : <DynamicWidget data={msg.content} />}
                    </div>
                  </div>
                ) : (
                  <div className="prose prose-stone prose-sm max-w-none prose-p:leading-relaxed prose-strong:text-inherit">
                    <ReactMarkdown>{typeof msg.content === "string" ? msg.content : ""}</ReactMarkdown>
                  </div>
                )}
              </div>
            </div>
          </div>
        ))}
        <div ref={(el) => { messagesEndRef.current = el; }} />
      </div>

      {/* Input Area - UNCONTROLLED COMPONENT (useRef) */}
      <div className="border-t border-gray-100 bg-white/80 p-4 backdrop-blur-md">
        <div className="relative flex items-center rounded-xl border border-gray-200 bg-white shadow-sm focus-within:border-emerald-400 focus-within:ring-4 focus-within:ring-emerald-100 transition-all duration-300">
          <input
            ref={inputRef}
            type="text"
            onKeyDown={(e) => e.key === "Enter" && handleSend()}
            placeholder="Tanya prediksi pertumbuhan faskes..."
            className="flex-1 border-none bg-transparent px-4 py-3.5 text-sm text-gray-800 placeholder-gray-400 focus:ring-0"
            disabled={isLoading}
          />
          <button
            onClick={handleSend}
            disabled={isLoading}
            className={`mr-2 rounded-lg p-2.5 transition-all duration-300 ${
              isLoading
                ? "cursor-not-allowed text-gray-300"
                : "bg-gradient-to-r from-emerald-500 to-teal-500 text-white shadow-lg shadow-emerald-500/30 hover:scale-105"
            }`}
          >
            {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
          </button>
        </div>
      </div>
    </div>
  );
}