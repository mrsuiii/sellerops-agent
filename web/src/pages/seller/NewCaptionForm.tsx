import { useState, type FormEvent } from "react";
import { useNavigate } from "react-router";
import { useCreateProposal, useCreateRequest, useSellerProducts } from "../../api/seller";
import type { ChangeField } from "../../api/types";
import { useToast } from "../../components/toast";
import { FIELD_LABEL } from "../../lib/format";
import { Spinner } from "./parts";

const FIELD_OPTIONS: ChangeField[] = ["name", "description", "tags"];

/** Ask the agent to write a caption from scratch. The agent may ask questions before it drafts. */
export function NewCaptionForm({ onDone }: { onDone: () => void }) {
  const products = useSellerProducts();
  const list = Object.values(products.data ?? {});
  const create = useCreateRequest();
  const navigate = useNavigate();
  const toast = useToast();
  const [productId, setProductId] = useState("");
  const [fields, setFields] = useState<ChangeField[]>(["description"]);
  const [instruction, setInstruction] = useState("");

  const submit = (event: FormEvent) => {
    event.preventDefault();
    if (!fields.length) {
      toast.show("Pilih minimal satu bagian caption", "error");
      return;
    }
    const product = productId || list[0]?.product_id;
    if (!product) return;
    create.mutate(
      { product_id: product, fields, instruction: instruction.trim() },
      {
        onSuccess: (view) => {
          const next = view.waiting_for === "decision" ? view.request?.proposal?.proposal_id : view.request_id;
          toast.show(
            view.waiting_for === "answer"
              ? "Agent punya pertanyaan untukmu"
              : view.state === "error"
                ? "Langkah agent gagal, coba lagi"
                : `Usulan ${next} siap diputuskan`,
          );
          setInstruction("");
          onDone();
          if (next) navigate(`/seller/${next}`);
        },
        onError: (error) => toast.show(error.message, "error"),
      },
    );
  };

  return (
    <div className="flex flex-col gap-8">
      <form onSubmit={submit} className="flex flex-col gap-5" aria-label="Minta agent menulis">
        <p className="text-sm text-muted">
          Agent menulis caption dari awal. Kalau ada fakta yang tidak ada di data produk (bahan, ukuran, perawatan), agent bertanya dulu
          kepadamu.
        </p>
        <label className="flex flex-col gap-1.5 text-sm">
          <span className="font-medium">Produk</span>
          <select className="field" value={productId || list[0]?.product_id || ""} onChange={(e) => setProductId(e.target.value)} required>
            {list.map((p) => (
              <option key={p.product_id} value={p.product_id}>
                {p.product_id} · {p.name}
              </option>
            ))}
          </select>
        </label>
        <fieldset className="flex flex-col gap-2 text-sm">
          <legend className="mb-2 font-medium">Bagian yang ditulis</legend>
          <div className="flex flex-wrap gap-2">
            {FIELD_OPTIONS.map((f) => {
              const checked = fields.includes(f);
              return (
                <label
                  key={f}
                  className={`inline-flex cursor-pointer items-center gap-2 rounded-full border px-3.5 py-1.5 transition ${
                    checked ? "border-ink bg-ink text-paper" : "border-line bg-surface hover:border-ink"
                  }`}
                >
                  <input
                    type="checkbox"
                    className="sr-only"
                    checked={checked}
                    onChange={(e) => setFields((cur) => (e.target.checked ? [...cur, f] : cur.filter((x) => x !== f)))}
                  />
                  {FIELD_LABEL[f]}
                </label>
              );
            })}
          </div>
        </fieldset>
        <label className="flex flex-col gap-1.5 text-sm">
          <span className="font-medium">Permintaan</span>
          <textarea
            rows={3}
            maxLength={1000}
            value={instruction}
            onChange={(e) => setInstruction(e.target.value)}
            placeholder="mis. fokus ke bahan dan cocok untuk kerja"
            className="field"
          />
        </label>
        <div className="flex items-center gap-3">
          <button className="btn-accent" type="submit" disabled={create.isPending || list.length === 0}>
            {create.isPending ? <Spinner label="Agent sedang membaca produk…" /> : "Minta agent menulis"}
          </button>
        </div>
      </form>

      <details className="panel group">
        <summary className="cursor-pointer px-5 py-3 text-sm font-medium">Atau tulis usulan sendiri (demo)</summary>
        <ManualProposal onDone={onDone} />
      </details>
    </div>
  );
}

function ManualProposal({ onDone }: { onDone: () => void }) {
  const products = useSellerProducts();
  const list = Object.values(products.data ?? {});
  const create = useCreateProposal();
  const navigate = useNavigate();
  const toast = useToast();

  const submit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const data = new FormData(event.currentTarget);
    const text = (key: string) => String(data.get(key) ?? "").trim();
    const body = {
      product_id: text("product_id"),
      reason: text("reason"),
      ...(text("name") && { name: text("name") }),
      ...(text("description") && { description: text("description") }),
      ...(text("tags") && { tags: text("tags").split(",").map((t) => t.trim()).filter(Boolean) }),
    };
    if (!body.name && !body.description && !body.tags) {
      toast.show("Isi minimal satu: nama, deskripsi, atau tag", "error");
      return;
    }
    create.mutate(body, {
      onSuccess: (created) => {
        toast.show(`Usulan ${created.proposal_id} dibuat`);
        event.currentTarget.reset();
        onDone();
        navigate(`/seller/${created.proposal_id}`);
      },
      onError: (error) => toast.show(error.message, "error"),
    });
  };

  return (
    <form onSubmit={submit} className="flex flex-col gap-4 border-t border-line p-5 text-sm">
      <p className="text-muted">Biasanya chat agent yang membuat usulan. Form ini untuk mencoba alurnya.</p>
      <label className="flex flex-col gap-1.5">
        <span className="font-medium">Produk</span>
        <select name="product_id" className="field" required>
          {list.map((p) => (
            <option key={p.product_id} value={p.product_id}>
              {p.product_id} · {p.name}
            </option>
          ))}
        </select>
      </label>
      <label className="flex flex-col gap-1.5">
        <span className="font-medium">Nama</span>
        <input name="name" maxLength={80} placeholder="(tidak diubah)" className="field" />
      </label>
      <label className="flex flex-col gap-1.5">
        <span className="font-medium">Deskripsi</span>
        <textarea name="description" rows={5} placeholder="(tidak diubah)" className="field" />
      </label>
      <label className="flex flex-col gap-1.5">
        <span className="font-medium">Tag</span>
        <input name="tags" placeholder="pisahkan dengan koma (tidak diubah)" className="field" />
      </label>
      <label className="flex flex-col gap-1.5">
        <span className="font-medium">Alasan</span>
        <input name="reason" required defaultValue="lebih menarik" className="field" />
      </label>
      <div>
        <button className="btn-ghost" type="submit" disabled={create.isPending || list.length === 0}>
          {create.isPending ? <Spinner label="Mengirim…" /> : "Kirim usulan"}
        </button>
      </div>
    </form>
  );
}
