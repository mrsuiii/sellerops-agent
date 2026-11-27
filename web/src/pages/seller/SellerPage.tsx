import { useState, type ReactNode } from "react";
import { Link, useParams } from "react-router";
import { ArrowLeft, Inbox, MessageSquare, Plus, Sparkles } from "lucide-react";
import { SellerChat } from "./SellerChat";
import { useLlmBudget, useProposalQueue, useRequestQueue, useSellerProducts } from "../../api/seller";
import { Drawer } from "../../components/drawer";
import { useSellerUser } from "../../lib/seller-user";
import { NewCaptionForm } from "./NewCaptionForm";
import { ProposalDetail } from "./ProposalDetail";
import { RequestDetail } from "./RequestDetail";
import { Notice, Spinner, StatusPill, fieldLabels } from "./parts";
import type { ChangeField } from "../../api/types";

const TABS = [
  { value: "pending", label: "Menunggu" },
  { value: "", label: "Semua" },
] as const;

export default function SellerPage() {
  const { id } = useParams();
  const [tab, setTab] = useState<string>("pending");
  const [creating, setCreating] = useState(false);
  const [chatOpen, setChatOpen] = useState(false);
  const { draft, user, setDraft } = useSellerUser();
  const budget = useLlmBudget();

  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-col gap-5">
        <div className="flex flex-col gap-2">
          <p className="eyebrow">Seller</p>
          <h1 className="font-display text-4xl leading-tight sm:text-5xl">Usulan caption</h1>
          <p className="max-w-2xl text-sm leading-relaxed text-muted">
            Agent mengusulkan perubahan, kamu yang memutuskan. Caption baru tampil di toko hanya setelah kamu setujui, dan bisa
            dikembalikan kapan saja.
          </p>
        </div>

        <div className="panel flex flex-col gap-3 p-3 sm:flex-row sm:items-center sm:justify-between sm:p-4">
          <div className="flex flex-col gap-1.5 sm:flex-row sm:items-center sm:gap-3">
            <label htmlFor="seller-name" className="text-sm text-muted">
              Masuk sebagai
            </label>
            <input
              id="seller-name"
              className={`field !w-full !py-2 sm:!w-48 ${user ? "" : "border-rose! focus:ring-rose/20!"}`}
              value={draft}
              maxLength={64}
              onChange={(e) => setDraft(e.target.value)}
              autoComplete="off"
              spellCheck={false}
              aria-invalid={!user}
              aria-describedby="seller-name-hint"
            />
            <span id="seller-name-hint" className={`text-xs ${user ? "text-muted" : "text-rose"}`}>
              {user ? "Tercatat di setiap perubahan" : "Pakai huruf, angka, titik, @, atau -"}
            </span>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            {budget.data?.enabled && (
              <span
                className="pill bg-accent-soft text-accent-strong"
                title="Demo publik: setiap tolak, jawab, atau minta caption memakai satu langkah agent. Setujui dan batalkan gratis."
              >
                Sisa agent jam ini: <strong className="tabular-nums">{budget.data.remaining_hour}/{budget.data.per_client_hour}</strong>
              </span>
            )}
            <button className="btn-ghost" onClick={() => setChatOpen(true)}>
              <MessageSquare size={16} aria-hidden /> Chat agent
            </button>
            <button className="btn-primary" onClick={() => setCreating(true)}>
              <Plus size={16} aria-hidden /> Caption baru
            </button>
          </div>
        </div>
      </header>

      <div className="grid gap-6 lg:grid-cols-[360px_1fr]">
        <Queue
          tab={tab}
          onTab={setTab}
          activeId={id}
          hideOnMobile={Boolean(id)}
          onNew={() => setCreating(true)}
          onChat={() => setChatOpen(true)}
        />

        <section className={`min-w-0 ${id ? "" : "hidden lg:block"}`} aria-live="polite">
          {id ? (
            <>
              <Link to="/seller" className="mb-4 inline-flex items-center gap-2 text-sm text-muted hover:text-ink lg:hidden">
                <ArrowLeft size={14} aria-hidden /> Antrean
              </Link>
              <div className="panel p-5 sm:p-8">
                {id.startsWith("CR-") ? <RequestDetail key={id} id={id} /> : <ProposalDetail key={id} id={id} />}
              </div>
            </>
          ) : (
            <div className="panel flex min-h-80 flex-col items-center justify-center gap-3 p-10 text-center">
              <span className="grid size-12 place-items-center rounded-full bg-accent-soft text-accent">
                <Sparkles size={22} aria-hidden />
              </span>
              <p className="font-display text-2xl">Pilih usulan di kiri</p>
              <p className="max-w-sm text-sm text-muted">
                Lihat perbedaannya kata per kata, lalu setujui, tolak dengan alasan, atau batalkan.
              </p>
            </div>
          )}
        </section>
      </div>

      <Drawer open={creating} onClose={() => setCreating(false)} title="Caption baru">
        <NewCaptionForm onDone={() => setCreating(false)} />
      </Drawer>

      <SellerChat open={chatOpen} onClose={() => setChatOpen(false)} />
      {!user && (
        <Notice tone="bad">Isi nama seller dulu (di atas) sebelum menyetujui atau menulis.</Notice>
      )}
    </div>
  );
}

function Queue({
  tab,
  onTab,
  activeId,
  hideOnMobile,
  onNew,
  onChat,
}: {
  tab: string;
  onTab: (tab: string) => void;
  activeId: string | undefined;
  hideOnMobile: boolean;
  onNew: () => void;
  onChat: () => void;
}) {
  const proposals = useProposalQueue(tab);
  const requests = useRequestQueue(tab);
  const products = useSellerProducts();
  const openRequests = (requests.data ?? []).filter((r) => r.status !== "proposed");
  const items = proposals.data ?? [];
  const loading = proposals.isPending || requests.isPending;
  const failed = proposals.isError || requests.isError;
  const empty = !loading && !failed && openRequests.length === 0 && items.length === 0;

  return (
    <aside className={`flex flex-col gap-3 ${hideOnMobile ? "hidden lg:flex" : ""}`} aria-label="Antrean">
      <div className="flex items-center gap-2" role="group" aria-label="Filter status">
        {TABS.map((item) => (
          <button
            key={item.value || "all"}
            type="button"
            aria-pressed={tab === item.value}
            onClick={() => onTab(item.value)}
            className={`rounded-full border px-3.5 py-1.5 text-sm transition ${
              tab === item.value ? "border-ink bg-ink text-paper" : "border-line bg-surface hover:border-ink"
            }`}
          >
            {item.label}
          </button>
        ))}
        {!loading && (
          <span className="ml-auto text-xs text-muted tabular-nums">{openRequests.length + items.length} item</span>
        )}
      </div>

      {loading && <Spinner label="Memuat antrean…" />}
      {failed && <Notice tone="bad">Antrean gagal dimuat. Coba muat ulang halaman.</Notice>}

      {empty && (
        <div className="panel flex flex-col items-center gap-3 p-8 text-center">
          <span className="grid size-11 place-items-center rounded-full bg-sand text-muted">
            <Inbox size={20} aria-hidden />
          </span>
          <p className="font-display text-xl">Antrean kosong</p>
          <p className="text-sm text-muted">
            {tab === "pending" ? "Tidak ada usulan yang menunggu." : "Belum ada usulan."} Minta agent menulis caption, atau tanya
            lewat chat.
          </p>
          <div className="mt-1 flex flex-wrap justify-center gap-2">
            <button className="btn-ghost !py-1.5" onClick={onChat}>
              Chat agent
            </button>
            <button className="btn-primary !py-1.5" onClick={onNew}>
              Caption baru
            </button>
          </div>
        </div>
      )}

      <ul className="flex flex-col gap-2">
        {openRequests.map((r) => {
          const product = products.data?.[r.product_id];
          return (
            <QueueItem
              key={r.request_id}
              to={`/seller/${r.request_id}`}
              active={activeId === r.request_id}
              thumbnail={product?.thumbnail}
              title={product?.name ?? r.product_id}
            >
              <StatusPill status={r.status} />
              <span className="mono">{r.request_id}</span>
              <span>· caption baru: {fieldLabels(r.fields as ChangeField[])}</span>
            </QueueItem>
          );
        })}
        {items.map((p) => {
          const product = products.data?.[p.product_id];
          return (
            <QueueItem
              key={p.proposal_id}
              to={`/seller/${p.proposal_id}`}
              active={activeId === p.proposal_id}
              thumbnail={product?.thumbnail}
              title={product?.name ?? p.product_id}
            >
              <StatusPill status={p.status} />
              <span className="mono">{p.proposal_id}</span>
              <span>· {fieldLabels(Object.keys(p.changes))}</span>
              {p.rejections_in_chain > 0 && <span>· {p.rejections_in_chain}/5 ditolak</span>}
            </QueueItem>
          );
        })}
      </ul>
    </aside>
  );
}

function QueueItem({
  to,
  active,
  thumbnail,
  title,
  children,
}: {
  to: string;
  active: boolean;
  thumbnail?: string;
  title: string;
  children: ReactNode;
}) {
  return (
    <li>
      <Link
        to={to}
        aria-current={active ? "true" : undefined}
        className={`flex gap-3 rounded-2xl border p-3 transition ${
          active ? "border-accent bg-accent-soft/50" : "border-line bg-surface hover:border-ink/40"
        }`}
      >
        {thumbnail ? (
          <img src={thumbnail} alt="" className="size-14 shrink-0 rounded-xl object-cover" />
        ) : (
          <span className="size-14 shrink-0 rounded-xl bg-sand" aria-hidden />
        )}
        <span className="flex min-w-0 flex-col gap-1">
          <span className="truncate text-sm font-medium">{title}</span>
          <span className="flex flex-wrap items-center gap-x-1.5 gap-y-1 text-xs text-muted">{children}</span>
        </span>
      </Link>
    </li>
  );
}
