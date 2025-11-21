"use client";

import { useEffect, useState } from "react";

export default function Loading() {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => {
      setMounted(true);
    }, 100000); // <- ubah durasi sesuka lo (ms)
    return () => clearTimeout(timer);
  }, []);

  return (
    <div
      className="flex min-h-screen items-center justify-center overflow-visible"
      style={{ backgroundColor: "#ffffff" }}
    >
      {/* Animated gradient orb background */}
      <div className="absolute inset-0 overflow-hidden">
        <div
          className="absolute h-96 w-96 rounded-full opacity-20 blur-3xl"
          style={{
            backgroundColor: "#44853B",
            top: "-10%",
            right: "-10%",
            animation: "float 12s ease-in-out infinite",
          }}
        />
        <div
          className="absolute h-96 w-96 rounded-full opacity-20 blur-3xl"
          style={{
            backgroundColor: "#2A4491",
            bottom: "-10%",
            left: "-10%",
            animation: "float 16s ease-in-out infinite reverse",
          }}
        />
      </div>

      {/* Main container */}
      <div className="relative z-10 flex flex-col items-center justify-center">
        {/* Animated logo container */}
        <div
          className="relative mb-8 overflow-visible"
          style={{
            animation: mounted ? "pulse-scale 4s ease-in-out infinite" : "none",
            overflow: "visible",
          }}
        >
          {/* Glowing circle backdrop */}
          <div
            className="absolute inset-0 rounded-full blur-2xl"
            style={{
              backgroundColor: "#44853B",
              opacity: 0.4,
              animation: "glow 5s ease-in-out infinite",
            }}
          />

          {/* Logo SVG (rings only) */}
          <svg
            width="120"
            height="120"
            viewBox="0 0 120 120"
            fill="none"
            className="relative z-10"
          >
            {/* Outer ring */}
            <circle
              cx="60"
              cy="60"
              r="55"
              stroke="#2A4491"
              strokeWidth="2"
              fill="none"
              opacity="0.8"
            />

            {/* Inner decorative circle */}
            <circle
              cx="60"
              cy="60"
              r="45"
              stroke="#44853B"
              strokeWidth="1.5"
              fill="none"
              opacity="0.6"
              style={{
                animation: "rotate-ring 20s linear infinite",
              }}
            />
          </svg>

          {/* Centered image logo (replace /logo.png with your logo path) */}
          <div className="absolute top-1/2 left-1/2 z-20 flex h-36 w-36 -translate-x-1/2 -translate-y-1/2 transform items-center justify-center overflow-hidden rounded-full bg-transparent">
            <img
              src="/logo.png"
              alt="Logo"
              className="h-full w-full object-contain"
              style={{ maxHeight: "none" }}
            />
          </div>
        </div>

        {/* Loading text */}
        <div className="mt-8 text-center">
          <h1
            className="mb-2 text-2xl font-bold"
            style={{
              color: "#2A4491",
              animation: "fade-in-up 0.8s ease-out forwards",
              opacity: mounted ? 1 : 0,
            }}
          >
            Loading
          </h1>

          {/* Animated dots */}
          <div className="flex items-center justify-center gap-1.5">
            {[0, 1, 2].map((i) => (
              <div
                key={i}
                className="h-2 w-2 rounded-full"
                style={{
                  backgroundColor: "#44853B",
                  animation: `bounce 2.8s ease-in-out ${i * 0.2}s infinite`,
                }}
              />
            ))}
          </div>
        </div>

        {/* Trademark text (reduced so it never appears larger than the logo) */}
        <p
          className="mt-6 font-light tracking-widest uppercase"
          style={{
            color: "#44853B",
            animation: "fade-in 1.2s ease-out forwards",
            opacity: mounted ? 1 : 0,
            fontSize: "10px", // explicitly smaller than the logo heading
            lineHeight: 1,
            letterSpacing: "0.12em",
          }}
        >
          developed by fwb
        </p>
      </div>

      <style>{`
        @keyframes pulse-scale {
          0%, 100% {
            transform: scale(1);
          }
          50% {
            transform: scale(1.05);
          }
        }

        @keyframes glow {
          0%, 100% {
            opacity: 0.3;
            filter: blur(20px);
          }
          50% {
            opacity: 0.6;
            filter: blur(30px);
          }
        }

        @keyframes rotate-ring {
          from {
            transform: rotate(0deg);
          }
          to {
            transform: rotate(360deg);
          }
        }

        @keyframes bounce {
          0%, 80%, 100% {
            transform: translateY(0);
            opacity: 0.6;
          }
          40% {
            transform: translateY(-8px);
            opacity: 1;
          }
        }

        @keyframes float {
          0%, 100% {
            transform: translateY(0px) translateX(0px);
          }
          50% {
            transform: translateY(20px) translateX(10px);
          }
        }

        @keyframes fade-in-up {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        @keyframes fade-in {
          from {
            opacity: 0;
          }
          to {
            opacity: 1;
          }
        }
      `}</style>
    </div>
  );
}
