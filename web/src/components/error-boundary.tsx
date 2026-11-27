import { Component, type ErrorInfo, type ReactNode } from "react";

type Props = { children: ReactNode };
type State = { error: Error | null };

/** Last line of defence: a render error shows a readable message instead of a blank page. */
export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("render error", error, info.componentStack);
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className="mx-auto flex min-h-dvh max-w-md flex-col items-center justify-center gap-4 px-6 text-center">
        <p className="font-display text-3xl">Ada yang tidak beres.</p>
        <p className="text-muted">Halaman gagal ditampilkan. Muat ulang untuk mencoba lagi.</p>
        <button className="btn-primary" onClick={() => window.location.reload()}>
          Muat ulang
        </button>
      </div>
    );
  }
}
