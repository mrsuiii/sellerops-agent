import { rupiah } from "../lib/format";

export function Price({ final, original, discount }: { final: number; original: number; discount: number }) {
  return (
    <span className="flex flex-wrap items-baseline gap-2">
      <strong className="font-semibold text-ink">{rupiah(final)}</strong>
      {discount >= 1 && <s className="text-sm text-muted">{rupiah(original)}</s>}
    </span>
  );
}
