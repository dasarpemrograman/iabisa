"use client";

import { useState } from "react";
import { ChevronDown } from "lucide-react";

const roles = ["Admin", "Manager", "User", "Viewer"];

export default function Navbar() {
  const [selectedRole, setSelectedRole] = useState("Admin");
  const [isOpen, setIsOpen] = useState(false);

  return (
    <nav
      className="flex h-16 items-center justify-between border-b px-6 shadow-sm"
      style={{
        background: "linear-gradient(90deg, #ffffff 0%, #e8f5e9 35%, #3d7d36 100%)",
        borderColor: "#dfe6dd",
      }}
    >
      {/* Left: Logo */}
      <div className="flex items-center gap-3">
        <div className="flex items-center rounded-xl py-1 shadow-sm">
          <img
            src="/logo.png"
            alt="Logo"
            className="h-1/2 w-1/2 object-contain"
          />
        </div>
      </div>

      {/* Role Dropdown */}
      <div className="relative">
        <button
          onClick={() => setIsOpen(!isOpen)}
          className="flex items-center gap-2 rounded-lg border px-4 py-2 text-sm font-medium shadow-sm transition-all hover:bg-white"
          style={{
            borderColor: "rgba(0, 0, 0, 0.1)",
            backgroundColor: "rgba(255, 255, 255, 0.9)",
            color: "#3d7d36",
          }}
        >
          {selectedRole}
          <ChevronDown size={16} className="opacity-70" />
        </button>

        {isOpen && (
          <div className="absolute right-0 z-50 mt-2 w-40 overflow-hidden rounded-lg border border-gray-200 bg-white shadow-md">
            {roles.map((role) => (
              <button
                key={role}
                onClick={() => {
                  setSelectedRole(role);
                  setIsOpen(false);
                }}
                className={`w-full px-4 py-2 text-left text-sm transition-colors ${
                  selectedRole === role
                    ? "bg-green-600 text-white"
                    : "text-gray-700 hover:bg-gray-100"
                }`}
              >
                {role}
              </button>
            ))}
          </div>
        )}
      </div>
    </nav>
  );
}
