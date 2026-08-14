/**
 * Render-error containment.
 *
 * A trading dashboard must **degrade, not disappear**. React unmounts the entire tree when
 * a render throws, so one unexpected field in one API payload is enough to replace the
 * whole page — kill switch, positions and all — with nothing at all. That failure mode is
 * far more dangerous than a wrong number: a blank page tells the operator nothing, and it
 * looks identical to a dead machine.
 *
 * Wrapping each panel means a malformed payload costs that one panel and no more.
 */

import { Component, type ErrorInfo, type ReactNode } from "react";

interface Props {
  children: ReactNode;
  /** Shown in the console trace so the failing panel is identifiable. */
  label: string;
}

interface State {
  message: string | null;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { message: null };
  }

  static getDerivedStateFromError(error: unknown): State {
    return { message: error instanceof Error ? error.message : String(error) };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    // Kept loud: a contained error is still a bug, and the operator's console is the only
    // place it can now be noticed.
    console.error(`[quantflow] "${this.props.label}" failed to render`, error, info.componentStack);
  }

  render(): ReactNode {
    const { message } = this.state;
    if (message !== null) {
      return (
        <div className="rounded border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-200">
          <span className="font-medium">Could not render this section.</span>{" "}
          <span className="text-amber-300/80">
            The API returned something unexpected: {message}
          </span>
        </div>
      );
    }
    return this.props.children;
  }
}
