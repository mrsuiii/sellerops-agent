import { useState, type FormEvent, type ReactNode } from "react";
import { useSearchParams } from "react-router";
import { ChevronLeft, ChevronRight, Search, SlidersHorizontal } from "lucide-react";
import { useCategories, useProducts } from "../../api/shop";
import type { ProductQuery } from "../../api/types";
import { ProductCard } from "../../components/product-card";
import { CATEGORY_LABEL } from "../../lib/format";

const PAGE_SIZE = 12;

export const SORTS: [string, string][] = [
  ["relevance", "Paling relevan"],
  ["price_asc", "Harga terendah"],
  ["price_desc", "Harga tertinggi"],
  ["rating", "Rating tertinggi"],
  ["discount", "Diskon terbesar"],
];

/** Filters live in the URL, so a filtered view can be shared and survives a reload. */
function readQuery(params: URLSearchParams): ProductQuery {
  return {
    q: params.get("q") ?? "",
    category: params.get("category") ?? "",
    audience: params.get("audience") ?? "",
    brand: params.get("brand") ?? "",
    max_price: params.get("max_price") ?? "",
    min_rating: params.get("min_rating") ?? "",
    in_stock: params.get("in_stock") === "1",
    sort: params.get("sort") ?? "relevance",
    page: Math.max(1, Number(params.get("page") ?? 1) || 1),
    page_size: PAGE_SIZE,
  };
}

export default function ShopPage() {
  const [params, setParams] = useSearchParams();
  const query = readQuery(params);
  const categories = useCategories();
  const products = useProducts(query);

  /** Changing a filter goes back to page 1; the page number is the only exception. */
  const update = (changes: Record<string, string | null>, resetPage = true) => {
    const next = new URLSearchParams(params);
    for (const [key, value] of Object.entries(changes)) {
      if (value === null || value === "") next.delete(key);
      else next.set(key, value);
    }
    if (resetPage) next.delete("page");
    setParams(next, { preventScrollReset: true });
  };

  const totalPages = products.data ? Math.max(1, Math.ceil(products.data.total / PAGE_SIZE)) : 1;
  const total = categories.data?.categories.reduce((sum, c) => sum + c.count, 0) ?? 0;

  return (
    <>
      <section className="relative min-w-0 overflow-hidden rounded-3xl bg-ink text-paper">
        <div className="relative z-10 grid min-w-0 grid-cols-1 gap-6 p-6 sm:p-12 md:grid-cols-[1.2fr_1fr] md:items-end">
          <div className="flex flex-col gap-4">
            <p className="eyebrow !text-[#e9a27f]">Koleksi musim ini</p>
            <h1 className="font-display text-4xl leading-[1.05] sm:text-5xl">
              Pakaian, sepatu, dan aksesori pilihan.
            </h1>
            <p className="max-w-md text-sm text-paper/70">
              Toko demo SellerOps. Data produk dari DummyJSON, harga dalam Rupiah.
            </p>
          </div>
          <SearchBox key={query.q} initial={query.q} onSearch={(q) => update({ q })} />
        </div>
        <div aria-hidden className="absolute -right-20 -bottom-24 size-72 rounded-full bg-accent/40 blur-3xl" />
      </section>

      <div className="mt-6 flex gap-2 overflow-x-auto pb-1" role="group" aria-label="Kategori">
        <CategoryChip label="Semua" count={total} active={!query.category} onClick={() => update({ category: null })} />
        {categories.data?.categories.map((c) => (
          <CategoryChip
            key={c.category}
            label={CATEGORY_LABEL[c.category] ?? c.category}
            count={c.count}
            active={query.category === c.category}
            onClick={() => update({ category: c.category })}
          />
        ))}
      </div>

      <div className="mt-8 grid min-w-0 grid-cols-1 gap-8 lg:grid-cols-[240px_1fr]">
        <Filters query={query} brands={categories.data?.brands ?? []} onChange={update} />

        <section aria-live="polite" className="min-w-0">
          <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
            <p className="text-sm text-muted">
              {products.data ? (
                <>
                  <strong className="text-ink">{products.data.total}</strong> produk
                  {query.q && (
                    <>
                      {" "}
                      untuk “<span className="text-ink">{query.q}</span>”
                    </>
                  )}
                </>
              ) : (
                "Memuat produk…"
              )}
            </p>
            <label className="flex items-center gap-2 text-sm text-muted lg:hidden">
              Urutkan
              <select className="field !w-auto !py-1.5" value={query.sort} onChange={(e) => update({ sort: e.target.value })}>
                {SORTS.map(([value, label]) => (
                  <option key={value} value={value}>
                    {label}
                  </option>
                ))}
              </select>
            </label>
          </div>

          {products.isError ? (
            <p className="panel p-6 text-center text-muted">Produk gagal dimuat. Coba muat ulang halaman.</p>
          ) : !products.data ? (
            <Skeleton />
          ) : products.data.items.length === 0 ? (
            <p className="panel p-10 text-center text-muted">Tidak ada produk yang cocok. Coba longgarkan filternya.</p>
          ) : (
            <div className={`grid grid-cols-2 gap-x-4 gap-y-8 xl:grid-cols-3 transition ${products.isPlaceholderData ? "opacity-60" : ""}`}>
              {products.data.items.map((product) => (
                <ProductCard key={product.product_id} product={product} />
              ))}
            </div>
          )}

          {products.data && totalPages > 1 && (
            <nav className="mt-10 flex items-center justify-center gap-2" aria-label="Halaman">
              <button
                className="btn-ghost size-9 !p-0"
                aria-label="Halaman sebelumnya"
                disabled={query.page <= 1}
                onClick={() => update({ page: String(query.page - 1) }, false)}
              >
                <ChevronLeft size={16} />
              </button>
              <span className="px-2 text-sm text-muted tabular-nums">
                {query.page} / {totalPages}
              </span>
              <button
                className="btn-ghost size-9 !p-0"
                aria-label="Halaman berikutnya"
                disabled={query.page >= totalPages}
                onClick={() => update({ page: String(query.page + 1) }, false)}
              >
                <ChevronRight size={16} />
              </button>
            </nav>
          )}
        </section>
      </div>
    </>
  );
}

function SearchBox({ initial, onSearch }: { initial: string; onSearch: (q: string) => void }) {
  const [value, setValue] = useState(initial);
  const submit = (event: FormEvent) => {
    event.preventDefault();
    onSearch(value.trim());
  };
  return (
    <form onSubmit={submit} role="search" className="relative">
      <Search size={18} className="pointer-events-none absolute top-1/2 left-4 -translate-y-1/2 text-muted" aria-hidden />
      <input
        type="search"
        aria-label="Cari produk"
        placeholder="Cari kemeja, sepatu, jam tangan…"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        className="w-full rounded-full border border-paper/15 bg-paper/10 py-3.5 pr-28 pl-11 text-sm text-paper placeholder:text-paper/50 focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent/40"
      />
      <button type="submit" className="btn-accent absolute top-1.5 right-1.5 !py-2">
        Cari
      </button>
    </form>
  );
}

function CategoryChip({ label, count, active, onClick }: { label: string; count: number; active: boolean; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={`inline-flex shrink-0 items-center gap-2 rounded-full border px-4 py-2 text-sm transition ${
        active ? "border-ink bg-ink text-paper" : "border-line bg-surface hover:border-ink"
      }`}
    >
      {label}
      <small className={active ? "text-paper/60" : "text-muted"}>{count}</small>
    </button>
  );
}

function Filters({
  query,
  brands,
  onChange,
}: {
  query: ProductQuery;
  brands: string[];
  onChange: (changes: Record<string, string | null>, resetPage?: boolean) => void;
}) {
  return (
    <aside className="hidden flex-col gap-5 lg:flex" aria-label="Filter">
      <div className="flex items-center gap-2 font-display text-lg">
        <SlidersHorizontal size={16} aria-hidden /> Filter
      </div>
      <FilterSelect label="Untuk" value={query.audience} onChange={(v) => onChange({ audience: v })}>
        <option value="">Semua</option>
        <option value="men">Pria</option>
        <option value="women">Wanita</option>
        <option value="unisex">Unisex</option>
      </FilterSelect>
      <FilterSelect label="Brand" value={query.brand} onChange={(v) => onChange({ brand: v })}>
        <option value="">Semua brand</option>
        {brands.map((b) => (
          <option key={b} value={b}>
            {b}
          </option>
        ))}
      </FilterSelect>
      <FilterSelect label="Harga maksimal" value={query.max_price} onChange={(v) => onChange({ max_price: v })}>
        <option value="">Tanpa batas</option>
        {[300_000, 500_000, 1_000_000, 2_500_000, 5_000_000].map((p) => (
          <option key={p} value={p}>
            Rp{p.toLocaleString("id-ID")}
          </option>
        ))}
      </FilterSelect>
      <FilterSelect label="Rating minimal" value={query.min_rating} onChange={(v) => onChange({ min_rating: v })}>
        <option value="">Semua</option>
        <option value="4">4+</option>
        <option value="4.5">4.5+</option>
      </FilterSelect>
      <label className="flex items-center gap-2 text-sm">
        <input
          type="checkbox"
          checked={query.in_stock}
          onChange={(e) => onChange({ in_stock: e.target.checked ? "1" : null })}
          className="size-4 accent-accent"
        />
        Hanya yang ada stok
      </label>
      <FilterSelect label="Urutkan" value={query.sort} onChange={(v) => onChange({ sort: v === "relevance" ? null : v }, false)}>
        {SORTS.map(([value, label]) => (
          <option key={value} value={value}>
            {label}
          </option>
        ))}
      </FilterSelect>
      <button
        type="button"
        className="btn-link justify-self-start text-left"
        onClick={() =>
          onChange({ audience: null, brand: null, max_price: null, min_rating: null, in_stock: null, sort: null, category: null, q: null })
        }
      >
        Reset filter
      </button>
    </aside>
  );
}

function FilterSelect({
  label,
  value,
  onChange,
  children,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  children: ReactNode;
}) {
  return (
    <label className="flex flex-col gap-1.5 text-sm">
      <span className="text-muted">{label}</span>
      <select className="field" value={value} onChange={(e) => onChange(e.target.value)}>
        {children}
      </select>
    </label>
  );
}

function Skeleton() {
  return (
    <div className="grid grid-cols-2 gap-x-4 gap-y-8 xl:grid-cols-3" aria-hidden>
      {Array.from({ length: 6 }, (_, i) => (
        <div key={i} className="animate-pulse">
          <div className="aspect-[4/5] rounded-2xl bg-sand" />
          <div className="mt-3 h-3 w-1/3 rounded bg-sand" />
          <div className="mt-2 h-4 w-3/4 rounded bg-sand" />
        </div>
      ))}
    </div>
  );
}
