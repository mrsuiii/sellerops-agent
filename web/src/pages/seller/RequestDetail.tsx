import { Link } from "react-router";
import {
  useAnswer,
  useAuditLog,
  useCancelRequest,
  useCaptionRequest,
  useProductFacts,
  useRequestApproval,
  useRetry,
  useSellerProducts,
} from "../../api/seller";
import { useToast } from "../../components/toast";
import { when, whoLabel } from "../../lib/format";
import { AuditTable, FactsPanel, Notice, QuestionForm, Spinner, StatusPill, fieldLabels } from "./parts";
import { useThreadResult } from "./use-thread-result";

export function RequestDetail({ id }: { id: string }) {
  const request = useCaptionRequest(id);
  const approval = useRequestApproval(id);
  const r = request.data;
  const products = useSellerProducts();
  const facts = useProductFacts(r?.product_id);
  const audit = useAuditLog(r?.product_id);
  const answer = useAnswer(id);
  const retry = useRetry(id);
  const cancel = useCancelRequest(id);
  const onResult = useThreadResult(id);
  const toast = useToast();

  if (request.isPending) return <Spinner label={`Memuat ${id}…`} />;
  if (request.isError || !r) return <Notice tone="bad">Gagal memuat {id}: {request.error?.message}</Notice>;

  const a = approval.data ?? null;
  const product = products.data?.[r.product_id];
  const next = a?.waiting_for === "decision" ? a.request?.proposal?.proposal_id : r.proposal_id;
  const busy = answer.isPending || retry.isPending || cancel.isPending;
  const fail = (error: Error) => toast.show(error.message, "error");

  return (
    <article className="flex flex-col gap-6">
      <header className="flex flex-col gap-4 sm:flex-row sm:items-start">
        {product?.thumbnail && <img src={product.thumbnail} alt="" className="size-24 shrink-0 rounded-2xl object-cover" />}
        <div className="flex min-w-0 flex-col gap-2">
          <h2 className="font-display text-3xl leading-tight">{product?.name ?? r.product_id}</h2>
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-sm text-muted">
            <StatusPill status={r.status} />
            <span className="mono">{r.request_id}</span>
            <span>· caption baru: {fieldLabels(r.fields)}</span>
            <span>· oleh {whoLabel(r.created_by)} · {when(r.created_at)}</span>
          </div>
        </div>
      </header>

      {a?.state === "error" && (
        <Notice tone="bad">
          <span className="flex flex-wrap items-center gap-3">
            Langkah agent gagal (mis. LLM tidak tersedia).
            <button className="btn-ghost !py-1" disabled={busy} onClick={() => retry.mutate(undefined, { onSuccess: (result) => onResult(result), onError: fail })}>
              {retry.isPending ? <Spinner label="Mencoba lagi…" /> : "Coba lagi"}
            </button>
          </span>
        </Notice>
      )}
      {(a?.stop_reason === "cancelled" || r.status === "cancelled") && <Notice>Permintaan dibatalkan.</Notice>}
      {next && (
        <Notice tone="ok">
          Agent sudah menulis usulan:{" "}
          <Link className="font-medium text-accent hover:underline" to={`/seller/${next}`}>
            {next} →
          </Link>
        </Notice>
      )}

      <section className="rounded-2xl bg-sand/60 p-5 text-[15px] leading-relaxed">
        <p className="mb-1 text-xs font-semibold tracking-wider text-muted uppercase">Permintaan</p>
        {r.instruction || <span className="text-muted">(tanpa instruksi khusus)</span>}
      </section>

      {a?.waiting_for === "answer" && a.request?.questions && (
        <QuestionForm
          questions={a.request.questions}
          error={typeof a.request.error === "string" ? a.request.error : null}
          pending={answer.isPending}
          onSubmit={(answers) => answer.mutate(answers, { onSuccess: (result) => onResult(result), onError: fail })}
        />
      )}

      {r.status === "open" && (
        <div>
          <button
            className="btn-danger"
            disabled={busy}
            onClick={() => cancel.mutate(undefined, { onSuccess: () => toast.show("Permintaan dibatalkan."), onError: fail })}
          >
            {cancel.isPending ? <Spinner label="Membatalkan…" /> : "Batalkan permintaan"}
          </button>
        </div>
      )}

      <FactsPanel facts={facts.data} />
      <AuditTable entries={audit.data ?? []} requestAware />
    </article>
  );
}
