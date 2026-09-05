import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Overridable so the dashboard can point at a non-default API port during development.
const API_TARGET = process.env.QF_API_URL ?? "http://localhost:8100";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Proxy the API in development so the browser sees one origin and CORS never enters
    // the picture locally.
    proxy: {
      // Every backend path the client touches. `/readyz` and `/metrics` sit at the root
      // rather than under /api so an orchestrator probe need not know the API version,
      // which means they have to be listed explicitly here.
      "/api": { target: API_TARGET, changeOrigin: true, ws: true },
      "/healthz": { target: API_TARGET, changeOrigin: true },
      "/readyz": { target: API_TARGET, changeOrigin: true },
      "/metrics": { target: API_TARGET, changeOrigin: true },
    },
  },
  test: {
    globals: true,
    environment: "jsdom",
    // jsdom has no ResizeObserver and no layout; without these shims every Recharts
    // container throws, the error boundaries contain it, and the suite stays green while
    // no chart is ever actually rendered.
    setupFiles: ["./src/test-setup.ts"],
  },
});
