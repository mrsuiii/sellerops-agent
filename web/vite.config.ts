import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// In dev, Vite serves the UI on :5173 and proxies the API to the FastAPI store on :8000
// (start it with `make store-fake` or `make store`). In production FastAPI serves dist/.
const API = process.env.SELLEROPS_API_URL ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      "/api": API,
      "/docs": API,
      "/openapi.json": API,
    },
  },
  build: {
    target: "es2022",
    sourcemap: false,
    reportCompressedSize: false,
  },
  test: {
    environment: "jsdom",
    include: ["src/**/*.test.{ts,tsx}"],
  },
});
