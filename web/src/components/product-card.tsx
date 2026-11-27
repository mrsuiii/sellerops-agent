import { Link } from "react-router";
import { Star } from "lucide-react";
import type { ProductSummary, Recommendation } from "../api/types";
import { Price } from "./price";

export function ProductCard({ product, reasons }: { product: ProductSummary; reasons?: Recommendation["reasons"] }) {
  const soldOut = product.stock <= 0;
  return (
    <Link
      to={`/p/${encodeURIComponent(product.product_id)}`}
      className="group flex flex-col gap-2 rounded-2xl outline-none focus-visible:ring-2 focus-visible:ring-accent"
    >
      <div className="relative aspect-[4/5] overflow-hidden rounded-2xl bg-sand">
        <img
          src={product.thumbnail}
          alt={product.name}
          loading="lazy"
          className="size-full object-cover transition duration-500 group-hover:scale-105"
        />
        {soldOut ? (
          <span className="pill absolute top-3 left-3 bg-ink text-paper">Habis</span>
        ) : product.discount_percent >= 1 ? (
          <span className="pill absolute top-3 left-3 bg-accent text-white">-{Math.round(product.discount_percent)}%</span>
        ) : null}
      </div>
      <div className="flex flex-col gap-0.5 px-0.5">
        <span className="text-xs tracking-wide text-muted uppercase">{product.brand}</span>
        <span className="line-clamp-2 text-sm font-medium leading-snug">{product.name}</span>
        <Price final={product.final_price_idr} original={product.price_idr} discount={product.discount_percent} />
        <span className="flex items-center gap-1 text-xs text-muted">
          <Star size={12} className="fill-accent text-accent" aria-hidden /> {product.rating.toFixed(1)}
        </span>
        {reasons && reasons.length > 0 && (
          <span className="mt-1 flex flex-wrap gap-1">
            {reasons.map((reason) => (
              <span key={reason} className="pill bg-sand text-muted">{reason}</span>
            ))}
          </span>
        )}
      </div>
    </Link>
  );
}
