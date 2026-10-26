import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The UI calls /api/*; Vite forwards it to the FastAPI server (no CORS needed in dev or preview).
const target = process.env.DOCSCOUT_API ?? "http://127.0.0.1:8000";
const proxy = { "/api": { target, changeOrigin: true, rewrite: (p: string) => p.replace(/^\/api/, "") } };

export default defineConfig({
  plugins: [react()],
  server: { port: 5173, strictPort: true, proxy },
  preview: { port: 4173, strictPort: true, proxy },
});
