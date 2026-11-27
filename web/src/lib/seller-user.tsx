import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from "react";
import { SELLER_USER_PATTERN, sellerUserStored } from "./api";

// There is no login yet: the seller picks a name and every write sends it as X-Seller-User.
// Demo only; the backend trusts the header.
const KEY = "kala-seller-user";
const DEFAULT_USER = "seller-demo";

type SellerUser = { draft: string; user: string; setDraft: (value: string) => void };
const Ctx = createContext<SellerUser | null>(null);

function initial(): string {
  return sellerUserStored() || DEFAULT_USER;
}

export function SellerUserProvider({ children }: { children: ReactNode }) {
  const [draft, setDraftState] = useState(initial);
  const setDraft = useCallback((value: string) => {
    setDraftState(value);
    try {
      localStorage.setItem(KEY, value.trim());
    } catch {
      /* storage unavailable: the name lasts for this page only */
    }
  }, []);
  const value = useMemo<SellerUser>(() => {
    const trimmed = draft.trim();
    return { draft, user: SELLER_USER_PATTERN.test(trimmed) ? trimmed : "", setDraft };
  }, [draft, setDraft]);
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useSellerUser(): SellerUser {
  const value = useContext(Ctx);
  if (!value) throw new Error("useSellerUser must be used inside SellerUserProvider");
  return value;
}
