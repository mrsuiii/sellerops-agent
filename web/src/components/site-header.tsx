import { useState } from "react";
import { Link, NavLink } from "react-router";
import { ShoppingBag } from "lucide-react";
import { useCart } from "../lib/cart";
import { CartDrawer } from "./cart-drawer";

export function SiteHeader() {
  const { count } = useCart();
  const [cartOpen, setCartOpen] = useState(false);
  const navClass = ({ isActive }: { isActive: boolean }) =>
    `rounded-full px-2.5 py-1.5 text-sm transition sm:px-3 ${isActive ? "bg-ink text-paper" : "text-muted hover:text-ink"}`;
  return (
    <>
      <header className="sticky top-0 z-40 border-b border-line/70 bg-paper/85 backdrop-blur-md">
        <div className="mx-auto flex max-w-7xl items-center justify-between gap-4 px-4 py-3 sm:px-6">
          <Link to="/" className="font-display text-xl tracking-tight" aria-label="Kala Studio, beranda">
            KALA<span className="text-accent">studio</span>
          </Link>
          <nav aria-label="Utama" className="flex items-center gap-0.5 sm:gap-1">
            <NavLink to="/" end className={navClass}>
              Toko
            </NavLink>
            <NavLink to="/seller" className={navClass}>
              Seller
            </NavLink>
            <button
              type="button"
              onClick={() => setCartOpen(true)}
              className="relative ml-2 inline-flex items-center gap-2 rounded-full border border-line bg-surface px-3 py-1.5 text-sm hover:border-ink"
              aria-label={`Keranjang, ${count} barang`}
            >
              <ShoppingBag size={16} aria-hidden />
              <span className="hidden sm:inline">Keranjang</span>
              <span className="tabular-nums" aria-hidden>
                {count}
              </span>
            </button>
          </nav>
        </div>
      </header>
      <CartDrawer open={cartOpen} onClose={() => setCartOpen(false)} />
    </>
  );
}
