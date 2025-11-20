"use client";

import type React from "react";

import { useState, useRef, useEffect } from "react";
import { Send, Maximize2, Minimize2 } from "lucide-react";

interface Message {
  id: string;
  text: string;
  sender: "user" | "assistant";
  timestamp: Date;
}

interface ChatbotPanelProps {
  fullscreen?: boolean;
  onFullscreen?: () => void;
  onExitFullscreen?: () => void;
}

export default function ChatbotPanel({
  fullscreen = false,
  onFullscreen,
  onExitFullscreen,
}: ChatbotPanelProps) {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      text: "Hello! How can I assist you today?",
      sender: "assistant",
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = () => {
    if (!input.trim()) return;

    // Add user message
    const userMessage: Message = {
      id: Date.now().toString(),
      text: input,
      sender: "user",
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    // Simulate assistant response
    setTimeout(() => {
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: "I understand. I'm here to help you with any questions or tasks.",
        sender: "assistant",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, assistantMessage]);
      setIsLoading(false);
    }, 500);
  };

  return (
    <div className="flex h-full flex-col bg-white">
      {/* Header */}
      <div className="flex h-14 flex-shrink-0 items-center justify-between border-b border-gray-200 px-4">
        <div>
          <h2 className="font-semibold text-gray-900">Assistant</h2>
          <p className="text-xs text-gray-500">Always available</p>
        </div>
        <button
          onClick={fullscreen ? onExitFullscreen : onFullscreen}
          className="rounded-lg p-1.5 transition-colors hover:bg-gray-100"
        >
          {fullscreen ? (
            <Minimize2 size={18} className="text-gray-600" />
          ) : (
            <Maximize2 size={18} className="text-gray-600" />
          )}
        </button>
      </div>

      {/* Messages Area */}
      <div className="flex-1 space-y-4 overflow-y-auto p-4">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex ${
              message.sender === "user" ? "justify-end" : "justify-start"
            } animate-in fade-in-0 slide-in-from-bottom-2 duration-300`}
          >
            <div
              className={`max-w-xs rounded-lg px-4 py-2 text-sm ${
                message.sender === "user"
                  ? "rounded-br-none text-white"
                  : "rounded-bl-none bg-gray-100 text-gray-900"
              }`}
              style={{
                background: message.sender === "user" ? "#44853B" : "#f3f4f6",
              }}
            >
              {message.text}
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="flex justify-start">
            <div className="flex items-center gap-1 rounded-lg bg-gray-100 px-4 py-2">
              <div className="h-2 w-2 animate-bounce rounded-full bg-gray-400" />
              <div className="h-2 w-2 animate-bounce rounded-full bg-gray-400 delay-100" />
              <div className="h-2 w-2 animate-bounce rounded-full bg-gray-400 delay-200" />
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="flex-shrink-0 border-t border-gray-200 p-4">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                handleSendMessage();
              }
            }}
            placeholder="Type your message..."
            className="flex-1 rounded-lg border border-gray-300 px-4 py-2 text-sm transition-colors focus:ring-2 focus:ring-offset-0 focus:outline-none"
            style={{ "--tw-ring-color": "#44853B" } as React.CSSProperties}
            disabled={isLoading}
          />
          <button
            onClick={handleSendMessage}
            disabled={isLoading || !input.trim()}
            className="flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium text-white transition-all hover:shadow-md disabled:cursor-not-allowed disabled:opacity-50"
            style={{ background: "#44853B" }}
          >
            <Send size={16} />
          </button>
        </div>
      </div>
    </div>
  );
}
