import { useEffect, type ReactNode } from "react";
import { X } from "lucide-react";

/** Side drawer with a backdrop; Escape and the backdrop close it. */
export function Drawer({
  open,
  onClose,
  title,
  children,
}: {
  open: boolean;
  onClose: () => void;
  title: string;
  children: ReactNode;
}) {
  useEffect(() => {
    if (!open) return;
    const onKey = (event: KeyboardEvent) => event.key === "Escape" && onClose();
    document.addEventListener("keydown", onKey);
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = "";
    };
  }, [open, onClose]);

  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50">
      <button aria-label="Tutup" className="absolute inset-0 bg-ink/30 backdrop-blur-[2px]" onClick={onClose} />
      <aside
        role="dialog"
        aria-modal="true"
        aria-label={title}
        className="absolute top-0 right-0 bottom-0 flex w-full max-w-md flex-col bg-paper shadow-2xl"
        style={{ animation: "rise 220ms ease-out" }}
      >
        <header className="flex items-center justify-between border-b border-line px-5 py-4">
          <h2 className="font-display text-xl">{title}</h2>
          <button className="rounded-full p-2 text-muted hover:bg-sand hover:text-ink" onClick={onClose} aria-label="Tutup">
            <X size={18} />
          </button>
        </header>
        <div className="flex-1 overflow-y-auto px-5 py-5">{children}</div>
      </aside>
    </div>
  );
}
