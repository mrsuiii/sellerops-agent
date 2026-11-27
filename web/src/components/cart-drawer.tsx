import { Link } from "react-router";
import { Minus, Plus } from "lucide-react";
import { cartActions, useCart } from "../lib/cart";
import { rupiah } from "../lib/format";
import { Drawer } from "./drawer";
import { useToast } from "./toast";

export function CartDrawer({ open, onClose }: { open: boolean; onClose: () => void }) {
  const { lines, total } = useCart();
  const toast = useToast();
  return (
    <Drawer open={open} onClose={onClose} title="Keranjang">
      {lines.length === 0 ? (
        <p className="py-12 text-center text-muted">Keranjang masih kosong.</p>
      ) : (
        <ul className="flex flex-col gap-4">
          {lines.map((line) => (
            <li key={line.product_id} className="flex gap-3 rounded-2xl border border-line bg-surface p-3">
              <img src={line.thumbnail} alt="" className="size-20 rounded-xl object-cover" />
              <div className="flex flex-1 flex-col gap-2">
                <Link to={`/p/${encodeURIComponent(line.product_id)}`} onClick={onClose} className="text-sm font-medium hover:text-accent">
                  {line.name}
                </Link>
                <div className="flex items-center gap-2">
                  <button className="btn-ghost size-8 !p-0" aria-label="Kurangi" onClick={() => cartActions.setQty(line.product_id, line.qty - 1)}>
                    <Minus size={14} />
                  </button>
                  <span className="w-6 text-center text-sm tabular-nums">{line.qty}</span>
                  <button
                    className="btn-ghost size-8 !p-0"
                    aria-label="Tambah"
                    disabled={line.qty >= line.stock}
                    onClick={() => cartActions.setQty(line.product_id, line.qty + 1)}
                  >
                    <Plus size={14} />
                  </button>
                </div>
              </div>
              <strong className="text-sm whitespace-nowrap">{rupiah(line.qty * line.final_price_idr)}</strong>
            </li>
          ))}
        </ul>
      )}
      {lines.length > 0 && (
        <div className="mt-6 flex flex-col gap-3 border-t border-line pt-5">
          <div className="flex justify-between text-sm">
            <span className="text-muted">Total</span>
            <strong className="text-base">{rupiah(total)}</strong>
          </div>
          <button
            className="btn-primary w-full"
            onClick={() => {
              cartActions.clear();
              onClose();
              toast.show("Pesanan demo dibuat. Tidak ada pembayaran sungguhan.");
            }}
          >
            Checkout (demo)
          </button>
        </div>
      )}
    </Drawer>
  );
}
