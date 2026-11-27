import { Link } from "react-router";

export default function NotFound({ title = "Halaman tidak ditemukan" }: { title?: string }) {
  return (
    <div className="flex flex-col items-center gap-4 py-24 text-center">
      <p className="eyebrow">404</p>
      <h1 className="font-display text-4xl">{title}</h1>
      <Link to="/" className="btn-primary mt-2">
        Kembali ke toko
      </Link>
    </div>
  );
}
