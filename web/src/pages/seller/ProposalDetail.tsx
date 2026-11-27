import { useState } from "react";
import { Link } from "react-router";
import {
  useAnswer,
  useSellerProducts,
  useAuditLog,
  useDecide,
  useProductFacts,
  useProposal,
  useProposalApproval,
  useRetry,
  useRevert,
  type Decision,
} from "../../api/seller";
import type { ChangeField } from "../../api/types";
import { useToast } from "../../components/toast";
import { FIELD_LABEL, when, whoLabel } from "../../lib/format";
import {
  AuditTable,
  ChangeBlock,
  FactsPanel,
  Notice,
  QuestionForm,
  SeenImages,
  Spinner,
  StatusPill,
  UsedFacts,
  fieldLabels,
} from "./parts";
import { STOP_REASON, useThreadResult } from "./use-thread-result";

const FIELDS: ChangeField[] = ["name", "description", "tags"];

export function ProposalDetail({ id }: { id: string }) {
  const proposal = useProposal(id);
  const approval = useProposalApproval(id);
  const p = proposal.data;
  const products = useSellerProducts();
  const facts = useProductFacts(p?.product_id);
  const audit = useAuditLog(p?.product_id);
  const decide = useDecide(id);
  const answer = useAnswer(id);
  const retry = useRetry(id);
  const revert = useRevert(id);
  const onResult = useThreadResult(id);
  const toast = useToast();
  const [rejecting, setRejecting] = useState(false);
  const [feedback, setFeedback] = useState("");
  const [addFields, setAddFields] = useState<ChangeField[]>([]);

  if (proposal.isPending) return <Spinner label={`Memuat ${id}…`} />;
  if (proposal.isError || !p) {
    return <Notice tone="bad">Gagal memuat {id}: {proposal.error?.message}</Notice>;
  }

  const a = approval.data ?? null;
  const product = products.data?.[p.product_id];
  const waitingHere = a?.waiting_for === "decision" && a.request?.proposal?.proposal_id === p.proposal_id;
  const askingHere = a?.waiting_for === "answer" && a.proposal_id === p.proposal_id;
  const fields = waitingHere ? (a?.request?.fields ?? []) : (Object.keys(p.changes) as ChangeField[]);
  const others = FIELDS.filter((f) => !fields.includes(f));
  const busy = decide.isPending || answer.isPending || retry.isPending || revert.isPending;

  const fail = (error: Error) => toast.show(error.message, "error");
  const doDecide = (body: Decision) =>
    decide.mutate(body, { onSuccess: (result) => onResult(result), onError: fail });

  const stop = a?.state === "done" && a.proposal_id === p.proposal_id ? STOP_REASON[a.stop_reason ?? ""] : undefined;
  const otherWaiting = a?.waiting_for === "decision" && !waitingHere ? a.request?.proposal?.proposal_id : undefined;

  return (
    <article className="flex flex-col gap-6">
      <header className="flex flex-col gap-4 sm:flex-row sm:items-start">
        {product?.thumbnail && <img src={product.thumbnail} alt="" className="size-24 shrink-0 rounded-2xl object-cover" />}
        <div className="flex min-w-0 flex-col gap-2">
          <h2 className="font-display text-3xl leading-tight">{product?.name ?? p.product_id}</h2>
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-sm text-muted">
            <StatusPill status={p.status} />
            <span className="mono">{p.proposal_id}</span>
            {p.revision_of && (
              <span>
                revisi dari{" "}
                <Link className="mono text-accent hover:underline" to={`/seller/${p.revision_of}`}>
                  {p.revision_of}
                </Link>
              </span>
            )}
            <span>· oleh {whoLabel(p.created_by)} · {when(p.created_at)}</span>
            <span>· {p.rejections_in_chain}/5 ditolak</span>
            <a className="text-accent hover:underline" href={`/p/${encodeURIComponent(p.product_id)}`} target="_blank" rel="noopener">
              lihat di toko ↗
            </a>
          </div>
        </div>
      </header>

      {otherWaiting && (
        <Notice>
          Versi ini sudah diputuskan. Yang menunggu sekarang:{" "}
          <Link className="font-medium text-accent hover:underline" to={`/seller/${otherWaiting}`}>
            {otherWaiting} →
          </Link>
        </Notice>
      )}
      {waitingHere && a?.request?.error && <Notice tone="bad">Jawaban terakhir ditolak: {a.request.error}</Notice>}
      {a?.state === "error" && (
        <Notice tone="bad">
          <span className="flex flex-wrap items-center gap-3">
            Langkah agent gagal (mis. LLM tidak tersedia). Penolakanmu sudah tercatat.
            <button className="btn-ghost !py-1" disabled={busy} onClick={() => retry.mutate(undefined, { onSuccess: (result) => onResult(result), onError: fail })}>
              {retry.isPending ? <Spinner label="Mencoba lagi…" /> : "Coba lagi"}
            </button>
          </span>
        </Notice>
      )}
      {stop && (
        <Notice tone={stop[0]}>
          {stop[1]}
          {a?.draft_error && <span className="mono ml-2 text-xs">{a.draft_error}</span>}
        </Notice>
      )}
      {p.status === "rejected" && (
        <Notice>
          Ditolak oleh {whoLabel(p.decided_by)}: {p.feedback ? `“${p.feedback}”` : "tanpa alasan"}
        </Notice>
      )}
      {p.status === "reverted" && (
        <Notice>
          Dikembalikan oleh {whoLabel(p.reverted_by)} · {when(p.reverted_at)}
        </Notice>
      )}

      <section className="rounded-2xl bg-sand/60 p-5 text-[15px] leading-relaxed">
        <p className="mb-1 text-xs font-semibold tracking-wider text-muted uppercase">
          Alasan {p.created_by.startsWith("agent") ? "agent" : "pengusul"}
        </p>
        {p.reason}
      </section>

      <SeenImages images={p.inputs?.images} />
      <UsedFacts facts={p.inputs?.seller_facts} />

      {askingHere && a?.request?.questions && (
        <QuestionForm
          questions={a.request.questions}
          error={typeof a.request.error === "string" ? a.request.error : null}
          pending={answer.isPending}
          onSubmit={(answers) => answer.mutate(answers, { onSuccess: (result) => onResult(result), onError: fail })}
        />
      )}

      <div className="flex flex-col gap-4">
        {(Object.entries(p.changes) as [ChangeField, NonNullable<(typeof p.changes)[ChangeField]>][]).map(([field, change]) => (
          <ChangeBlock key={field} field={field} change={change} inScope={fields.includes(field)} />
        ))}
      </div>

      {waitingHere && (
        <div className="flex flex-col gap-4">
          <div className="flex flex-wrap items-center gap-3">
            <button className="btn-accent" disabled={busy} onClick={() => doDecide({ decision: "approve" })}>
              {decide.isPending && decide.variables?.decision === "approve" ? <Spinner label="Menerapkan caption…" /> : "Setujui & terapkan"}
            </button>
            <button className="btn-ghost" disabled={busy} onClick={() => setRejecting((open) => !open)} aria-expanded={rejecting}>
              Tolak…
            </button>
            <button className="btn-danger" disabled={busy} onClick={() => doDecide({ decision: "cancel" })}>
              {decide.isPending && decide.variables?.decision === "cancel" ? <Spinner label="Membatalkan…" /> : "Batalkan"}
            </button>
          </div>

          {rejecting && (
            <form
              className="panel flex flex-col gap-4 p-5"
              onSubmit={(event) => {
                event.preventDefault();
                doDecide({ decision: "reject", feedback: feedback.trim(), add_fields: addFields });
              }}
            >
              <label className="flex flex-col gap-1.5 text-sm">
                <span className="font-medium">Apa yang perlu diubah? Agent akan menulis revisi.</span>
                <textarea
                  rows={3}
                  maxLength={1000}
                  value={feedback}
                  onChange={(e) => setFeedback(e.target.value)}
                  placeholder="mis. terlalu panjang, sebut bahannya"
                  className="field"
                  autoFocus
                />
              </label>
              {others.length > 0 && (
                <fieldset className="flex flex-wrap items-center gap-3 text-sm">
                  <legend className="mb-2 text-muted">Boleh juga ubah:</legend>
                  {others.map((f) => (
                    <label key={f} className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        className="size-4 accent-accent"
                        checked={addFields.includes(f)}
                        onChange={(e) => setAddFields((cur) => (e.target.checked ? [...cur, f] : cur.filter((x) => x !== f)))}
                      />
                      {FIELD_LABEL[f]}
                    </label>
                  ))}
                </fieldset>
              )}
              <p className="text-xs text-muted">
                Revisi hanya boleh mengubah: {fieldLabels(fields)}
                {others.length > 0 && " (+ yang dicentang)"}.
              </p>
              <div>
                <button className="btn-primary" type="submit" disabled={busy}>
                  {decide.isPending && decide.variables?.decision === "reject" ? <Spinner label="Agent sedang menulis revisi…" /> : "Tolak & minta revisi"}
                </button>
              </div>
            </form>
          )}
        </div>
      )}

      {p.status === "applied" && (
        <div className="flex flex-wrap items-center gap-3">
          <button
            className="btn-danger"
            disabled={busy}
            onClick={() =>
              revert.mutate(undefined, {
                onSuccess: () => toast.show("Caption sebelumnya tampil lagi di toko."),
                onError: fail,
              })
            }
          >
            {revert.isPending ? <Spinner label="Mengembalikan caption…" /> : "Kembalikan caption sebelumnya"}
          </button>
          <span className="text-xs text-muted">Hanya bisa selama toko masih menampilkan caption dari usulan ini.</span>
        </div>
      )}

      <FactsPanel facts={facts.data} />
      <AuditTable entries={audit.data ?? []} />
    </article>
  );
}
