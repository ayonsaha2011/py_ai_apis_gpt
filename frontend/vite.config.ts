import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const gateway = process.env.VITE_GATEWAY_PROXY ?? "http://127.0.0.1:8080";
const apiPrefixes = ["/health", "/status", "/auth", "/v1", "/rag", "/history", "/admin"];

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
  server: {
    proxy: Object.fromEntries(apiPrefixes.map((prefix) => [prefix, { target: gateway, changeOrigin: true }])),
  },
});
