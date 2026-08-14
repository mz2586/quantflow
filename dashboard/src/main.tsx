import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import { ErrorBoundary } from "./components/ErrorBoundary";
import "./index.css";

const container = document.getElementById("root");
if (!container) throw new Error("#root is missing from index.html");

createRoot(container).render(
  <StrictMode>
    {/* Last-resort net. The per-panel boundaries should catch everything before this. */}
    <ErrorBoundary label="QuantFlow dashboard">
      <App />
    </ErrorBoundary>
  </StrictMode>,
);
