import { createContext, useCallback, useContext, useMemo, useRef, useState, type ReactNode } from "react";

type Tone = "info" | "error";
type ToastState = { id: number; text: string; tone: Tone } | null;
type ToastApi = { show: (text: string, tone?: Tone) => void };

const Ctx = createContext<ToastApi | null>(null);

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toast, setToast] = useState<ToastState>(null);
  const timer = useRef<number | undefined>(undefined);

  const show = useCallback((text: string, tone: Tone = "info") => {
    window.clearTimeout(timer.current);
    setToast({ id: Date.now(), text, tone });
    timer.current = window.setTimeout(() => setToast(null), tone === "error" ? 5000 : 3200);
  }, []);

  const api = useMemo(() => ({ show }), [show]);
  return (
    <Ctx.Provider value={api}>
      {children}
      <div aria-live="polite" className="pointer-events-none fixed inset-x-0 bottom-4 z-[60] flex justify-center px-4">
        {toast && (
          <div
            key={toast.id}
            role={toast.tone === "error" ? "alert" : "status"}
            className={`pointer-events-auto max-w-md rounded-full px-5 py-3 text-sm font-medium shadow-lg ${
              toast.tone === "error" ? "bg-rose text-white" : "bg-ink text-paper"
            }`}
            style={{ animation: "rise 200ms ease-out" }}
          >
            {toast.text}
          </div>
        )}
      </div>
    </Ctx.Provider>
  );
}

export function useToast(): ToastApi {
  const value = useContext(Ctx);
  if (!value) throw new Error("useToast must be used inside ToastProvider");
  return value;
}
