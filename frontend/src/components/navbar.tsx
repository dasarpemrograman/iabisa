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
        background: "linear-gradient(90deg, #44853B 0%, #3a6f34 100%)",
        borderColor: "#2d5a28",
      }}
    >
      {/* Logo */}
      <div className="flex items-center gap-3">
        <div
          className="flex h-8 w-8 items-center justify-center rounded-lg text-sm font-bold text-white shadow-md"
          style={{ background: "#2d5a28" }}
        >
          DB
        </div>
        <span className="text-lg font-semibold text-white">Dashboard</span>
      </div>

      {/* Role Dropdown */}
      <div className="relative">
        <button
          onClick={() => setIsOpen(!isOpen)}
          className="flex items-center gap-2 rounded-lg border bg-white px-4 py-2 transition-colors hover:bg-gray-100"
          style={{
            borderColor: "rgba(255, 255, 255, 0.3)",
            color: "#44853B",
          }}
        >
          <span className="text-sm font-medium">{selectedRole}</span>
          <ChevronDown size={16} />
        </button>

        {isOpen && (
          <div className="absolute right-0 z-50 mt-2 w-40 rounded-lg border border-gray-200 bg-white shadow-lg">
            {roles.map((role) => (
              <button
                key={role}
                onClick={() => {
                  setSelectedRole(role);
                  setIsOpen(false);
                }}
                className={`w-full px-4 py-2 text-left text-sm transition-colors ${
                  selectedRole === role
                    ? "text-white"
                    : "text-gray-700 hover:bg-gray-50"
                }`}
                style={{
                  background: selectedRole === role ? "#44853B" : "transparent",
                }}
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
