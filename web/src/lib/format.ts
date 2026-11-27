export const rupiah = (amount: number): string => `Rp${amount.toLocaleString("id-ID")}`;

export const when = (iso: string | null | undefined): string =>
  iso ? new Date(iso).toLocaleString("id-ID", { dateStyle: "medium", timeStyle: "short" }) : "";

export const CATEGORY_LABEL: Record<string, string> = {
  bags: "Tas",
  dresses: "Dress",
  fragrances: "Parfum",
  jewellery: "Perhiasan",
  shirts: "Kemeja",
  shoes: "Sepatu",
  sunglasses: "Kacamata",
  tops: "Atasan",
  watches: "Jam tangan",
};

export const FIELD_LABEL: Record<string, string> = {
  name: "Nama",
  description: "Deskripsi",
  tags: "Tag",
};

export const STATUS_LABEL: Record<string, string> = {
  pending: "menunggu",
  applied: "diterapkan",
  rejected: "ditolak",
  cancelled: "dibatalkan",
  stale: "basi",
  reverted: "dikembalikan",
  open: "diproses agent",
  proposed: "sudah diusulkan",
};

/** Who an actor is, in Indonesian. `seller:` prefixes are stripped; the agent is named. */
export const whoLabel = (actor: string | null | undefined): string =>
  actor === "agent:caption-approval" ? "agent (revisi)" : String(actor ?? "").replace(/^seller:/, "");
