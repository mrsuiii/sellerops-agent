import { useState } from "react";
import { Link, useParams } from "react-router";
import { ArrowLeft, ShieldCheck, Star, Truck } from "lucide-react";
import { useProduct, useSimilar } from "../../api/shop";
import { ProductCard } from "../../components/product-card";
import { Price } from "../../components/price";
import { useToast } from "../../components/toast";
import { cartActions } from "../../lib/cart";
import { CATEGORY_LABEL } from "../../lib/format";
import NotFound from "../NotFound";

export default function ProductPage() {
  const { id = "" } = useParams();
  const product = useProduct(id);
  const similar = useSimilar(id);
  const toast = useToast();
  const [active, setActive] = useState(0);

  if (product.isError) {
    const missing = product.error instanceof Error && "status" in product.error && product.error.status === 404;
    return missing ? <NotFound title="Produk tidak ditemukan" /> : <p className="panel p-6 text-muted">Produk gagal dimuat.</p>;
  }
  if (!product.data) return <ProductSkeleton />;

  const p = product.data;
  const images = p.images.length ? p.images : [p.thumbnail];
  const soldOut = p.stock <= 0;
  const addToCart = () => {
    cartActions.add({
      product_id: p.product_id,
      name: p.name,
      thumbnail: p.thumbnail,
      final_price_idr: p.final_price_idr,
      stock: p.stock,
    });
    toast.show(`${p.name} masuk keranjang`);
  };

  return (
    <article>
      <Link to="/" className="inline-flex items-center gap-2 text-sm text-muted hover:text-ink">
        <ArrowLeft size={14} aria-hidden /> Kembali ke katalog
      </Link>

      <div className="mt-6 grid gap-10 md:grid-cols-2">
        <div className="flex flex-col gap-3">
          <div className="overflow-hidden rounded-3xl bg-sand">
            <img src={images[active] ?? images[0]} alt={p.name} className="aspect-[4/5] size-full object-cover" />
          </div>
          {images.length > 1 && (
            <div className="grid grid-cols-5 gap-2">
              {images.map((src, i) => (
                <button
                  key={src}
                  type="button"
                  onClick={() => setActive(i)}
                  aria-label={`Foto ${i + 1}`}
                  aria-current={i === active}
                  className={`overflow-hidden rounded-xl border-2 transition ${i === active ? "border-accent" : "border-transparent"}`}
                >
                  <img src={src} alt="" className="aspect-square size-full object-cover" />
                </button>
              ))}
            </div>
          )}
        </div>

        <div className="flex flex-col gap-5">
          <div>
            <p className="text-xs tracking-[0.18em] text-muted uppercase">
              {p.brand} · {CATEGORY_LABEL[p.category] ?? p.category}
            </p>
            <h1 className="mt-2 font-display text-4xl leading-tight">{p.name}</h1>
          </div>
          <div className="text-2xl">
            <Price final={p.final_price_idr} original={p.price_idr} discount={p.discount_percent} />
          </div>
          <p className="flex items-center gap-1.5 text-sm text-muted">
            <Star size={14} className="fill-accent text-accent" aria-hidden /> {p.rating.toFixed(1)} · {p.reviews.length} ulasan
          </p>
          <p className="leading-relaxed text-ink/80">{p.description}</p>

          <button type="button" className="btn-primary w-full !py-3.5 sm:w-auto sm:px-10" disabled={soldOut} onClick={addToCart}>
            {soldOut ? "Stok habis" : "Tambah ke keranjang"}
          </button>

          <ul className="grid gap-3 text-sm sm:grid-cols-2">
            <li className="panel flex items-start gap-3 p-4">
              <ShieldCheck size={18} className="mt-0.5 shrink-0 text-sage" aria-hidden />
              <span>
                <strong className="block font-medium">Garansi</strong>
                {p.warranty ?? "-"}
              </span>
            </li>
            <li className="panel flex items-start gap-3 p-4">
              <Truck size={18} className="mt-0.5 shrink-0 text-sage" aria-hidden />
              <span>
                <strong className="block font-medium">Pengiriman</strong>
                {p.shipping ?? "-"}
                {p.return_policy && <span className="block text-muted">{p.return_policy}</span>}
              </span>
            </li>
          </ul>

          <dl className="grid grid-cols-[auto_1fr] gap-x-6 gap-y-2 text-sm">
            <dt className="text-muted">Stok</dt>
            <dd>{soldOut ? "Habis" : `${p.stock} tersedia`}</dd>
            <dt className="text-muted">SKU</dt>
            <dd className="mono">{p.sku ?? p.product_id}</dd>
          </dl>
        </div>
      </div>

      <section className="mt-16">
        <h2 className="font-display text-2xl">Ulasan</h2>
        {p.reviews.length === 0 ? (
          <p className="mt-4 text-sm text-muted">Belum ada ulasan.</p>
        ) : (
          <ul className="mt-5 grid gap-4 md:grid-cols-2">
            {p.reviews.map((r, i) => (
              <li key={`${r.reviewer}-${i}`} className="panel p-5">
                <p className="flex items-center gap-1 text-sm">
                  <Star size={13} className="fill-accent text-accent" aria-hidden /> {r.rating}
                </p>
                <p className="mt-2 text-sm leading-relaxed">{r.comment}</p>
                <p className="mt-3 text-xs text-muted">
                  {r.reviewer} · {r.date}
                </p>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="mt-16">
        <h2 className="font-display text-2xl">Mungkin kamu suka</h2>
        {similar.isError ? (
          <p className="mt-4 text-sm text-muted">Rekomendasi belum tersedia.</p>
        ) : !similar.data ? (
          <p className="mt-4 text-sm text-muted">Memuat rekomendasi…</p>
        ) : similar.data.length === 0 ? (
          <p className="mt-4 text-sm text-muted">Belum ada rekomendasi.</p>
        ) : (
          <div className="mt-6 grid grid-cols-2 gap-x-4 gap-y-8 lg:grid-cols-4">
            {similar.data.map((r) => (
              <ProductCard key={r.product_id} product={r} reasons={r.reasons} />
            ))}
          </div>
        )}
      </section>
    </article>
  );
}

function ProductSkeleton() {
  return (
    <div className="grid animate-pulse gap-10 md:grid-cols-2" aria-hidden>
      <div className="aspect-[4/5] rounded-3xl bg-sand" />
      <div className="flex flex-col gap-4">
        <div className="h-4 w-1/3 rounded bg-sand" />
        <div className="h-10 w-3/4 rounded bg-sand" />
        <div className="h-6 w-1/4 rounded bg-sand" />
      </div>
    </div>
  );
}
