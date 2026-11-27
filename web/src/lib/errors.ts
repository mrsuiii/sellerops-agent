// Backend messages that are technical or English. Shown as plain Indonesian; anything else
// passes through unchanged (the API already writes readable messages for business errors).
const MESSAGES: [RegExp, string][] = [
  [/^no .* with id/i, "Data ini tidak ditemukan. Mungkin id-nya salah atau sudah dihapus."],
  [/unknown product_id/i, "Produk tidak ditemukan."],
  [/internal server error/i, "Terjadi kesalahan di server. Coba lagi sebentar."],
];

export function friendly(message: string | undefined): string {
  if (!message) return "Terjadi kesalahan. Coba lagi.";
  return MESSAGES.find(([pattern]) => pattern.test(message))?.[1] ?? message;
}
