// Thin fetch wrapper for the FastAPI backend. Every write sends X-Seller-User:
// there is no login yet, so the seller name is a trusted header (demo only).

export class ApiError extends Error {
  readonly status: number;
  readonly code: string | undefined;

  constructor(status: number, body: unknown) {
    const detail = (body as { detail?: unknown } | null)?.detail;
    super(typeof detail === "string" ? detail : `HTTP ${status}`);
    this.status = status;
    this.code = (body as { code?: string } | null)?.code;
  }
}

export const SELLER_USER_PATTERN = /^[\w.@-]{1,64}$/;

export function sellerUserStored(): string {
  try {
    return localStorage.getItem("kala-seller-user") ?? "";
  } catch {
    return "";
  }
}

type Json = Record<string, unknown> | unknown[];

export async function api<T>(
  path: string,
  options: { method?: "GET" | "POST"; body?: Json; seller?: string; signal?: AbortSignal } = {},
): Promise<T> {
  const { method = "GET", body, seller, signal } = options;
  const headers: Record<string, string> = {};
  if (method !== "GET") {
    if (!seller || !SELLER_USER_PATTERN.test(seller)) {
      throw new ApiError(0, { detail: "isi nama seller dulu (kanan atas)" });
    }
    headers["X-Seller-User"] = seller;
    headers["Content-Type"] = "application/json";
  }
  let response: Response;
  try {
    response = await fetch(path, {
      method,
      headers,
      body: body === undefined ? undefined : JSON.stringify(body),
      signal,
    });
  } catch {
    throw new ApiError(0, { detail: "Tidak bisa terhubung ke server. Coba lagi sebentar." });
  }
  const data: unknown = await response.json().catch(() => null);
  if (!response.ok) throw new ApiError(response.status, data);
  return data as T;
}
