// Cart lives in the browser (localStorage). Checkout is a demo: no payment, no order.
import { useSyncExternalStore } from "react";

export type CartLine = {
  product_id: string;
  name: string;
  thumbnail: string;
  final_price_idr: number;
  stock: number;
  qty: number;
};

type Lines = Record<string, CartLine>;
const KEY = "kala-cart";
const listeners = new Set<() => void>();
const EMPTY: Lines = {};

function read(): Lines {
  try {
    const parsed: unknown = JSON.parse(localStorage.getItem(KEY) ?? "{}");
    return parsed && typeof parsed === "object" ? (parsed as Lines) : {};
  } catch {
    return {};
  }
}

// useSyncExternalStore needs a stable snapshot: keep the same object until the cart changes.
let snapshot: Lines = read();

function write(next: Lines): void {
  snapshot = next;
  try {
    localStorage.setItem(KEY, JSON.stringify(next));
  } catch {
    /* storage unavailable (private mode): the cart still works for this page view */
  }
  listeners.forEach((listener) => listener());
}

export const cartActions = {
  add(product: Omit<CartLine, "qty">): void {
    const line = snapshot[product.product_id] ?? { ...product, qty: 0 };
    write({ ...snapshot, [product.product_id]: { ...line, qty: Math.min(line.qty + 1, product.stock) } });
  },
  setQty(productId: string, qty: number): void {
    const line = snapshot[productId];
    if (!line) return;
    const next = { ...snapshot };
    if (qty <= 0) delete next[productId];
    else next[productId] = { ...line, qty: Math.min(qty, line.stock) };
    write(next);
  },
  clear(): void {
    write({});
  },
};

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function useCart(): { lines: CartLine[]; count: number; total: number } {
  const lines = useSyncExternalStore(subscribe, () => snapshot, () => EMPTY);
  const values = Object.values(lines);
  return {
    lines: values,
    count: values.reduce((sum, line) => sum + line.qty, 0),
    total: values.reduce((sum, line) => sum + line.qty * line.final_price_idr, 0),
  };
}
