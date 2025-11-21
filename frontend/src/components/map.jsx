"use client";

import React, { useLayoutEffect, useRef } from "react";
import * as am5 from "@amcharts/amcharts5";
import * as amcharts_map from "@amcharts/amcharts5/map";
import am5themes_Animated from "@amcharts/amcharts5/themes/Animated";
import indonesiaHigh from "@amcharts/amcharts5-geodata/indonesiaHigh";

export default function BPJSIndonesiaMap() {
  const chartRef = useRef(null);

  useLayoutEffect(() => {
    if (!chartRef.current) return;

    const root = am5.Root.new(chartRef.current);
    root.setThemes([am5themes_Animated.new(root)]);

    const chart = root.container.children.push(
      amcharts_map.MapChart.new(root, {
        panX: "translateX",
        panY: "translateY",
        projection: amcharts_map.geoMercator(),
      }),
    );

    const polygonSeries = chart.series.push(
      amcharts_map.MapPolygonSeries.new(root, {
        geoJSON: indonesiaHigh,
      }),
    );

    polygonSeries.mapPolygons.template.setAll({
      tooltipText: "{name}\n{value}",
      templateField: "polygonSettings",
      interactive: true,
      stroke: am5.color(0xffffff),
      strokeWidth: 1,
    });

    polygonSeries.mapPolygons.template.states.create("hover", {
      fill: am5.color(0x677935),
    });

    // Data dummy pengguna BPJS per provinsi
    const data = [
      { id: "ID-JK", value: 12000000 },
      { id: "ID-JB", value: 15000000 },
      { id: "ID-JT", value: 13000000 },
      { id: "ID-JI", value: 10000000 },
      { id: "ID-BT", value: 4000000 },
      { id: "ID-YO", value: 3000000 },
      { id: "ID-SS", value: 8000000 },
      { id: "ID-BA", value: 5000000 },
      // Tambahkan provinsi lain sesuai kebutuhan
    ];

    // Hitung min/max
    const values = data.map((d) => d.value);
    const minVal = Math.min(...values);
    const maxVal = Math.max(...values);

    // Fungsi interpolate warna: hijau → kuning → merah
    function getColor(value) {
      const t = (value - minVal) / (maxVal - minVal); // 0..1
      // Linear interpolate: hijau (0x00ff00) → kuning (0xffff00) → merah (0xff0000)
      let r, g, b;
      if (t < 0.5) {
        // hijau → kuning
        const ratio = t / 0.5;
        r = Math.round(0 + ratio * 255); // 0 → 255
        g = 255;
      } else {
        // kuning → merah
        const ratio = (t - 0.5) / 0.5;
        r = 255;
        g = Math.round(255 - ratio * 255); // 255 → 0
      }
      b = 0;
      return am5.color((r << 16) + (g << 8) + b);
    }

    // Assign color
    polygonSeries.data.setAll(
      data.map((d) => ({
        id: d.id,
        value: d.value,
        polygonSettings: { fill: getColor(d.value) },
      })),
    );

    chart.appear(1000, 100);

    return () => {
      root.dispose();
    };
  }, []);

  return <div ref={chartRef} style={{ width: "100%", height: "600px" }} />;
}
