import { useState, type FormEvent, type ReactNode } from "react";
import { wordDiff, type Segment } from "../../lib/diff";
import { FIELD_LABEL, STATUS_LABEL, when, whoLabel } from "../../lib/format";
import type { AuditEntry, ChangeField, Fact, FieldChange, Question } from "../../api/types";
import { Link } from "react-router";

// ---------- small presentational pieces ----------

export function Spinner({ label }: { label?: string }) {
  return (
    <span className="inline-flex items-center gap-2 text-sm text-muted" role="status">
      <span className="size-3.5 animate-spin rounded-full border-2 border-accent border-t-transparent" aria-hidden />
      {label}
    </span>
  );
}

const TONES = {
  neutral: "bg-sand text-ink",
  ok: "bg-sage-soft text-sage",
  bad: "bg-rose-soft text-rose",
  warn: "bg-amber-soft text-[#7a5200]",
  info: "bg-accent-soft text-accent-strong",
} as const;
export type Tone = keyof typeof TONES;

export function Notice({ tone = "neutral", children }: { tone?: Tone; children: ReactNode }) {
  return <div className={`rounded-2xl px-4 py-3 text-sm leading-relaxed ${TONES[tone]}`}>{children}</div>;
}

const STATUS_TONE: Record<string, Tone> = {
  pending: "warn",
  open: "info",
  applied: "ok",
  rejected: "bad",
  stale: "bad",
  cancelled: "neutral",
  reverted: "neutral",
  proposed: "neutral",
};

export function StatusPill({ status }: { status: string }) {
  return <span className={`pill ${TONES[STATUS_TONE[status] ?? "neutral"]}`}>{STATUS_LABEL[status] ?? status}</span>;
}

export const fieldLabels = (fields: ChangeField[] | string[]) => fields.map((f) => FIELD_LABEL[f] ?? f).join(", ");

// ---------- diff ----------

function Segments({ segments, side }: { segments: Segment[]; side: "old" | "new" }) {
  return (
    <>
      {segments.map((segment, i) =>
        segment.kind === "same" ? (
          <span key={i}>{segment.text}</span>
        ) : segment.kind === "removed" ? (
          <del key={i} className="rounded bg-rose-soft px-0.5 text-rose decoration-rose/60">
            {segment.text}
          </del>
        ) : (
          <ins key={i} className="rounded bg-sage-soft px-0.5 text-sage no-underline">
            {segment.text}
          </ins>
        ),
      )}
      {side === "new" && segments.length === 0 && <span className="text-muted">(kosong)</span>}
    </>
  );
}

function TagSide({ tags, other, side }: { tags: string[]; other: string[]; side: "old" | "new" }) {
  const otherSet = new Set(other);
  if (tags.length === 0) return <span className="text-muted">(kosong)</span>;
  return (
    <span className="flex flex-wrap gap-1.5">
      {tags.map((tag) => {
        const same = otherSet.has(tag);
        const tone = same ? "bg-sand text-ink" : side === "old" ? "bg-rose-soft text-rose line-through" : "bg-sage-soft text-sage";
        return (
          <span key={tag} className={`pill ${tone}`}>
            {tag}
          </span>
        );
      })}
    </span>
  );
}

export function ChangeBlock({ field, change, inScope }: { field: ChangeField; change: FieldChange; inScope: boolean }) {
  const isTags = field === "tags";
  const oldList = isTags ? ((change.old as string[] | null) ?? []) : [];
  const newList = isTags ? ((change.new as string[] | null) ?? []) : [];
  const diff = isTags ? null : wordDiff(String(change.old ?? ""), String(change.new ?? ""));
  const count = (value: unknown) => (isTags ? `${(value as string[] | null)?.length ?? 0} tag` : `${String(value ?? "").length} karakter`);

  return (
    <section className="panel overflow-hidden">
      <header className="flex items-center justify-between border-b border-line px-5 py-3">
        <h3 className="font-medium">
          {FIELD_LABEL[field] ?? field}
          {!inScope && <small className="ml-2 font-normal text-muted">(di luar cakupan revisi)</small>}
        </h3>
      </header>
      <div className="grid md:grid-cols-2">
        <div className="flex flex-col gap-2 p-5 md:border-r md:border-line">
          <span className="text-xs font-semibold tracking-wider text-muted uppercase">Sekarang</span>
          <p className="text-[15px] leading-relaxed break-words">
            {isTags ? <TagSide tags={oldList} other={newList} side="old" /> : diff && <Segments segments={diff.old} side="old" />}
          </p>
          <span className="text-xs text-muted">{count(change.old)}</span>
        </div>
        <div className="flex flex-col gap-2 bg-sage-soft/30 p-5">
          <span className="text-xs font-semibold tracking-wider text-sage uppercase">Usulan</span>
          <p className="text-[15px] leading-relaxed break-words">
            {isTags ? <TagSide tags={newList} other={oldList} side="new" /> : diff && <Segments segments={diff.new} side="new" />}
          </p>
          <span className="text-xs text-muted">{count(change.new)}</span>
        </div>
      </div>
    </section>
  );
}

// ---------- agent context ----------

export function SeenImages({ images }: { images?: { sha256: string; source_url?: string; width: number; height: number }[] }) {
  if (!images?.length) return null;
  return (
    <section className="flex flex-col gap-3">
      <h3 className="text-sm font-medium">Gambar yang dilihat model</h3>
      <div className="flex flex-wrap gap-3">
        {images.map((img) => (
          <a
            key={img.sha256}
            href={`/api/seller/images/${img.sha256}`}
            target="_blank"
            rel="noopener"
            title={img.source_url}
            className="group flex w-32 flex-col gap-1.5"
          >
            <img
              src={`/api/seller/images/${img.sha256}`}
              alt=""
              loading="lazy"
              className="aspect-square w-32 rounded-xl border border-line object-cover transition group-hover:border-ink"
            />
            <span className="mono text-[11px] text-muted">
              {img.sha256.slice(0, 8)} · {img.width}×{img.height}
            </span>
          </a>
        ))}
      </div>
    </section>
  );
}

export function UsedFacts({ facts }: { facts?: Record<string, string | null> }) {
  const entries = Object.entries(facts ?? {});
  if (!entries.length) return null;
  return (
    <section className="flex flex-col gap-2">
      <h3 className="text-sm font-medium">Fakta dari seller yang dipakai</h3>
      <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 rounded-2xl bg-sand/60 p-4 text-sm">
        {entries.map(([key, value]) => (
          <div key={key} className="contents">
            <dt className="mono text-muted">{key}</dt>
            <dd>{value === null ? <em className="text-muted">seller tidak tahu</em> : value}</dd>
          </div>
        ))}
      </dl>
    </section>
  );
}

export function QuestionForm({
  questions,
  error,
  pending,
  onSubmit,
}: {
  questions: Question[];
  error?: string | null;
  pending: boolean;
  onSubmit: (answers: Record<string, string | null>) => void;
}) {
  const submit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const data = new FormData(event.currentTarget);
    const answers = Object.fromEntries(
      questions.map((q) => {
        const raw = String(data.get(q.key) ?? "").trim();
        return [q.key, raw === "" ? null : raw];
      }),
    );
    onSubmit(answers);
  };
  return (
    <form onSubmit={submit} className="panel flex flex-col gap-4 border-accent/40 bg-accent-soft/40 p-5" aria-label="Pertanyaan agent">
      <div>
        <p className="font-display text-xl">Agent butuh info dulu sebelum menulis caption</p>
        <p className="mt-1 text-sm text-muted">
          Jawabanmu disimpan sebagai fakta produk, jadi agent tidak perlu bertanya lagi lain kali. Yang dikosongkan dicatat sebagai
          “tidak tahu”: caption akan menyebut bahwa data tambahan dibutuhkan.
        </p>
      </div>
      {error && <Notice tone="bad">Jawaban ditolak: {error}</Notice>}
      {questions.map((q) => (
        <label key={q.key} className="flex flex-col gap-1.5 text-sm">
          <span className="font-medium">{q.question}</span>
          <textarea name={q.key} rows={2} maxLength={300} placeholder="kosongkan kalau tidak tahu" className="field" />
          <span className="mono text-[11px] text-muted">{q.key}</span>
        </label>
      ))}
      <div>
        <button className="btn-accent" type="submit" disabled={pending}>
          {pending ? <Spinner label="Agent sedang menulis…" /> : "Kirim jawaban"}
        </button>
      </div>
    </form>
  );
}

export function FactsPanel({ facts }: { facts?: Record<string, Fact> }) {
  const entries = Object.entries(facts ?? {});
  if (!entries.length) return null;
  return (
    <Disclosure title={`Yang sudah kamu ceritakan ke agent (${entries.length})`}>
      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm">
          <thead className="text-xs text-muted">
            <tr>
              <th className="py-2 pr-4 font-medium">Fakta</th>
              <th className="py-2 pr-4 font-medium">Jawaban</th>
              <th className="py-2 pr-4 font-medium">Oleh</th>
              <th className="py-2 font-medium">Dari</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-line">
            {entries.map(([key, fact]) => (
              <tr key={key}>
                <td className="mono py-2 pr-4 align-top">{key}</td>
                <td className="py-2 pr-4 align-top">{fact.value === null ? <em className="text-muted">tidak tahu</em> : fact.value}</td>
                <td className="py-2 pr-4 align-top text-muted">
                  {whoLabel(fact.by)} · {when(fact.at)}
                </td>
                <td className="mono py-2 align-top">
                  {fact.source ? <Link className="text-accent hover:underline" to={`/seller/${fact.source}`}>{fact.source}</Link> : "langsung"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Disclosure>
  );
}

export function AuditTable({ entries, requestAware = false }: { entries: AuditEntry[]; requestAware?: boolean }) {
  return (
    <Disclosure title="Riwayat produk (audit log)">
      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm">
          <thead className="text-xs text-muted">
            <tr>
              <th className="py-2 pr-4 font-medium">Waktu</th>
              <th className="py-2 pr-4 font-medium">Oleh</th>
              <th className="py-2 pr-4 font-medium">Aksi</th>
              <th className="py-2 font-medium">Usulan</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-line">
            {entries.map((entry, i) => (
              <tr key={`${entry.at}-${i}`}>
                <td className="py-2 pr-4 align-top text-muted">{when(entry.at)}</td>
                <td className="py-2 pr-4 align-top">{whoLabel(entry.actor)}</td>
                <td className="py-2 pr-4 align-top">{entry.action}</td>
                <td className="mono py-2 align-top">
                  {entry.proposal_id ? (
                    <Link className="text-accent hover:underline" to={`/seller/${entry.proposal_id}`}>
                      {entry.proposal_id}
                    </Link>
                  ) : requestAware ? (
                    entry.detail?.request_id ?? ""
                  ) : (
                    ""
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Disclosure>
  );
}

export function Disclosure({ title, children }: { title: string; children: ReactNode }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="panel">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="flex w-full items-center justify-between px-5 py-3 text-left text-sm font-medium"
      >
        {title}
        <span className={`text-muted transition ${open ? "rotate-90" : ""}`} aria-hidden>
          ›
        </span>
      </button>
      {open && <div className="border-t border-line px-5 py-4">{children}</div>}
    </div>
  );
}
