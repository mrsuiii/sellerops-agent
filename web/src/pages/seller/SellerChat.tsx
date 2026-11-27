import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent, type ReactNode } from "react";
import { Link } from "react-router";
import { Plus, Send } from "lucide-react";
import { useChatHistory, useSendChat, type ChatReply, type ChatTurnMessage } from "../../api/chat";
import { Drawer } from "../../components/drawer";
import { useToast } from "../../components/toast";
import { useChatThread } from "../../lib/chat-thread";
import { Spinner } from "./parts";

const TOOL_LABEL: Record<string, string> = {
  get_sales_summary: "penjualan",
  get_product_performance: "produk",
  get_inventory_status: "stok",
  get_reviews: "ulasan",
  get_product: "data produk",
  propose_caption_update: "usulan caption",
};
const MAX_CHARS = 2000;
const SUGGESTIONS = [
  "Berapa penjualan 7 hari terakhir?",
  "Produk apa yang stoknya paling cepat habis?",
  "Perbaiki deskripsi FS180 agar lebih menarik",
];

/** Chat with the seller agent (real model in production). Replies can read the store and propose caption changes. */
export function SellerChat({ open, onClose }: { open: boolean; onClose: () => void }) {
  const { threadId, startNew } = useChatThread();
  const history = useChatHistory(threadId, open);
  const send = useSendChat(threadId);
  const toast = useToast();
  const [draft, setDraft] = useState("");
  const [pending, setPending] = useState<string | null>(null);
  const [lastReply, setLastReply] = useState<ChatReply | null>(null);
  const endRef = useRef<HTMLDivElement>(null);

  // Keep the newest message in view.
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [history.data, pending, open]);

  const submit = (event?: FormEvent) => {
    event?.preventDefault();
    const text = draft.trim();
    if (!text || send.isPending) return;
    setDraft("");
    setPending(text);
    send.mutate(text, {
      onSuccess: (reply) => setLastReply(reply),
      onError: (error) => {
        setDraft(text);
        toast.show(error.message, "error");
      },
      onSettled: () => setPending(null),
    });
  };

  const onKey = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
  };

  const messages: ChatTurnMessage[] = history.data ?? [];
  const lastAgentIndex = messages.map((m) => m.role).lastIndexOf("agent");

  return (
    <Drawer open={open} onClose={onClose} title="Chat dengan agent">
      <div className="flex h-full min-h-[60vh] flex-col gap-4">
        <div className="flex items-center justify-between gap-3 text-sm">
          <p className="text-muted">Tanya soal penjualan, stok, atau ulasan. Minta perbaikan caption, lalu setujui di antrean.</p>
          <button
            type="button"
            className="btn-ghost shrink-0 !py-1.5"
            onClick={() => {
              startNew();
              setLastReply(null);
            }}
            disabled={send.isPending}
          >
            <Plus size={14} aria-hidden /> Baru
          </button>
        </div>

        <div className="flex flex-1 flex-col gap-3 overflow-y-auto rounded-2xl border border-line bg-surface p-4" aria-live="polite">
          {history.isPending && open && <Spinner label="Memuat percakapan…" />}
          {history.isSuccess && messages.length === 0 && !pending && (
            <div className="m-auto flex max-w-xs flex-col items-center gap-3 text-center">
              <p className="text-sm text-muted">Belum ada percakapan. Coba salah satu:</p>
              {SUGGESTIONS.map((text) => (
                <button
                  key={text}
                  type="button"
                  onClick={() => setDraft(text)}
                  className="w-full rounded-xl border border-line bg-paper px-3 py-2 text-left text-sm transition hover:border-ink"
                >
                  {text}
                </button>
              ))}
            </div>
          )}

          {messages.map((message, i) => (
            <Bubble key={i} message={message}>
              {i === lastAgentIndex && lastReply && <TurnDetails reply={lastReply} />}
            </Bubble>
          ))}

          {pending && <Bubble message={{ role: "user", text: pending }} />}
          {pending && (
            <div className="self-start">
              <Spinner label="Agent sedang berpikir…" />
            </div>
          )}
          <div ref={endRef} />
        </div>

        <form onSubmit={submit} className="flex items-end gap-2">
          <label className="flex-1">
            <span className="sr-only">Pesan untuk agent</span>
            <textarea
              value={draft}
              onChange={(e) => setDraft(e.target.value.slice(0, MAX_CHARS))}
              onKeyDown={onKey}
              rows={2}
              placeholder="Tulis pertanyaan… (Enter untuk kirim, Shift+Enter baris baru)"
              className="field resize-none"
              disabled={send.isPending}
            />
          </label>
          <button className="btn-accent !px-4" type="submit" disabled={send.isPending || !draft.trim()} aria-label="Kirim">
            <Send size={16} aria-hidden />
          </button>
        </form>
      </div>
    </Drawer>
  );
}

function Bubble({ message, children }: { message: ChatTurnMessage; children?: ReactNode }) {
  const mine = message.role === "user";
  return (
    <div className={`flex max-w-[88%] flex-col gap-2 ${mine ? "self-end items-end" : "self-start items-start"}`}>
      <div
        className={`rounded-2xl px-4 py-2.5 text-[15px] leading-relaxed break-words whitespace-pre-wrap ${
          mine ? "rounded-br-md bg-ink text-paper" : "rounded-bl-md bg-sand text-ink"
        }`}
      >
        {message.text}
      </div>
      {children}
    </div>
  );
}

function TurnDetails({ reply }: { reply: ChatReply }) {
  if (reply.tools.length === 0 && reply.proposals.length === 0) return null;
  return (
    <div className="flex flex-wrap gap-1.5 text-xs">
      {reply.tools.map((tool, i) => (
        <span
          key={`${tool.name}-${i}`}
          className={`pill ${tool.ok ? "bg-surface text-muted ring-1 ring-line" : "bg-rose-soft text-rose"}`}
        >
          {tool.ok ? "dicek" : "gagal"}: {TOOL_LABEL[tool.name] ?? tool.name}
        </span>
      ))}
      {reply.proposals.map((id) => (
        <Link key={id} to={`/seller/${id}`} className="pill bg-accent-soft text-accent-strong hover:underline">
          Usulan {id} →
        </Link>
      ))}
    </div>
  );
}
