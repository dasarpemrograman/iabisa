"use client";

import React, { useLayoutEffect, useRef } from "react";
import * as am5 from "@amcharts/amcharts5";
import * as amcharts_map from "@amcharts/amcharts5/map";
import am5themes_Animated from "@amcharts/amcharts5/themes/Animated";
import indonesiaHigh from "@amcharts/amcharts5-geodata/indonesiaHigh";

// --- HELPER: Normalize Province Names to ISO IDs ---
const PROVINCE_TO_ID = {
  ACEH: "ID-AC",
  "SUMATERA UTARA": "ID-SU",
  "NORTH SUMATRA": "ID-SU",
  "SUMATERA BARAT": "ID-SB",
  "WEST SUMATRA": "ID-SB",
  RIAU: "ID-RI",
  JAMBI: "ID-JA",
  "SUMATERA SELATAN": "ID-SS",
  "SOUTH SUMATRA": "ID-SS",
  BENGKULU: "ID-BE",
  LAMPUNG: "ID-LA",
  "KEPULAUAN BANGKA BELITUNG": "ID-BB",
  "BANGKA BELITUNG": "ID-BB",
  "KEPULAUAN RIAU": "ID-KR",
  "RIAU ISLANDS": "ID-KR",
  "DKI JAKARTA": "ID-JK",
  JAKARTA: "ID-JK",
  "JAWA BARAT": "ID-JB",
  "WEST JAVA": "ID-JB",
  "JAWA TENGAH": "ID-JT",
  "CENTRAL JAVA": "ID-JT",
  "DI YOGYAKARTA": "ID-YO",
  YOGYAKARTA: "ID-YO",
  "JAWA TIMUR": "ID-JI",
  "EAST JAVA": "ID-JI",
  BANTEN: "ID-BT",
  BALI: "ID-BA",
  "NUSA TENGGARA BARAT": "ID-NB",
  "WEST NUSA TENGGARA": "ID-NB",
  "NUSA TENGGARA TIMUR": "ID-NT",
  "EAST NUSA TENGGARA": "ID-NT",
  "KALIMANTAN BARAT": "ID-KB",
  "WEST KALIMANTAN": "ID-KB",
  "KALIMANTAN TENGAH": "ID-KT",
  "CENTRAL KALIMANTAN": "ID-KT",
  "KALIMANTAN SELATAN": "ID-KS",
  "SOUTH KALIMANTAN": "ID-KS",
  "KALIMANTAN TIMUR": "ID-KI",
  "EAST KALIMANTAN": "ID-KI",
  "KALIMANTAN UTARA": "ID-KU",
  "NORTH KALIMANTAN": "ID-KU",
  "SULAWESI UTARA": "ID-SA",
  "NORTH SULAWESI": "ID-SA",
  "SULAWESI TENGAH": "ID-ST",
  "CENTRAL SULAWESI": "ID-ST",
  "SULAWESI SELATAN": "ID-SN",
  "SOUTH SULAWESI": "ID-SN",
  "SULAWESI TENGGARA": "ID-SG",
  "SOUTHEAST SULAWESI": "ID-SG",
  GORONTALO: "ID-GO",
  "SULAWESI BARAT": "ID-SR",
  "WEST SULAWESI": "ID-SR",
  MALUKU: "ID-MA",
  "MALUKU UTARA": "ID-MU",
  "NORTH MALUKU": "ID-MU",
  "PAPUA BARAT": "ID-PB",
  "WEST PAPUA": "ID-PB",
  PAPUA: "ID-PA",
};

function normalizeName(name) {
  if (!name) return "";
  // Normalize text: remove dots, trim, uppercase
  return name.toString().toUpperCase().replace(/\./g, "").trim();
}

export default function BPJSIndonesiaMap({ data }) {
  const chartRef = useRef(null);

  useLayoutEffect(() => {
    if (!chartRef.current) return;

    // 1. PROCESS DATA
    const finalData = [];
    let rows = [];
    let provKey = "";
    let valKey = "";

    // Robustly handle different data shapes (from Backend or Raw Array)
    if (data && !Array.isArray(data) && data.data) {
      rows = data.data;
      provKey = data.province_key;
      valKey = data.value_key;
    } else if (Array.isArray(data)) {
      rows = data;
      const keys = rows.length > 0 ? Object.keys(rows[0]) : [];
      // Guess columns if not provided
      provKey = keys.find((k) => k.toLowerCase().includes("prov")) || keys[0];
      valKey = keys.find((k) => k !== provKey) || keys[1];
    }

    if (Array.isArray(rows)) {
      rows.forEach((row) => {
        const rawName = row[provKey];
        const val = row[valKey];
        const id = PROVINCE_TO_ID[normalizeName(rawName)];
        if (id) {
          finalData.push({ id, value: Number(val) || 0, name: rawName });
        }
      });
    }

    // 2. INITIALIZE MAP
    const root = am5.Root.new(chartRef.current);
    root.setThemes([am5themes_Animated.new(root)]);

    const chart = root.container.children.push(
      amcharts_map.MapChart.new(root, {
        panX: "translateX",
        panY: "translateY",
        projection: amcharts_map.geoMercator(),
        layout: root.horizontalLayout,
      }),
    );

    const polygonSeries = chart.series.push(
      amcharts_map.MapPolygonSeries.new(root, {
        geoJSON: indonesiaHigh,
        valueField: "value",
        calculateAggregates: true,
      }),
    );

    // 3. STYLING
    polygonSeries.mapPolygons.template.setAll({
      tooltipText: "{name}: {value}",
      interactive: true,
      stroke: am5.color(0xffffff),
      strokeWidth: 1,
      fill: am5.color(0xdadada), // Default gray for no data
    });

    polygonSeries.mapPolygons.template.states.create("hover", {
      fill: am5.color(0x677935),
    });

    // 4. HEATMAP RULES
    if (finalData.length > 0) {
      polygonSeries.set("heatRules", [
        {
          target: polygonSeries.mapPolygons.template,
          dataField: "value",
          min: am5.color(0x8ab4f8), // Light Blue
          max: am5.color(0x174ea6), // Deep Blue
          key: "fill",
        },
      ]);

      polygonSeries.data.setAll(finalData);
    }

    // 5. LEGEND (Optional, improves UX)
    if (finalData.length > 0) {
      const heatLegend = chart.children.push(
        am5.HeatLegend.new(root, {
          orientation: "vertical",
          startColor: am5.color(0x8ab4f8),
          endColor: am5.color(0x174ea6),
          startText: "Low",
          endText: "High",
          stepCount: 5,
        }),
      );

      heatLegend.startLabel.setAll({
        fontSize: 12,
        fill: heatLegend.get("startColor"),
      });
      heatLegend.endLabel.setAll({
        fontSize: 12,
        fill: heatLegend.get("endColor"),
      });

      heatLegend.set("x", am5.percent(100));
      heatLegend.set("centerX", am5.percent(100));
      heatLegend.set("y", am5.percent(90));
      heatLegend.set("centerY", am5.percent(100));
    }

    chart.appear(1000, 100);

    return () => {
      root.dispose();
    };
  }, [data]);

  // FIX: Set width/height to 100% so it fills the Dashboard Tile
  return (
    <div
      ref={chartRef}
      style={{
        width: "100%",
        height: "100%",
        minHeight: "300px", // Prevents collapse on very small screens
        position: "relative",
      }}
    />
  );
}
