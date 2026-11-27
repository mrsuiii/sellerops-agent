import { Route, Routes } from "react-router";
import { SiteHeader } from "./components/site-header";
import NotFound from "./pages/NotFound";
import ProductPage from "./pages/shop/ProductPage";
import ShopPage from "./pages/shop/ShopPage";
import SellerPage from "./pages/seller/SellerPage";

export default function App() {
  return (
    <div className="flex min-h-dvh flex-col">
      <SiteHeader />
      <main className="mx-auto w-full max-w-7xl flex-1 px-4 pt-6 pb-20 sm:px-6 sm:pt-10">
        <Routes>
          <Route index element={<ShopPage />} />
          <Route path="p/:id" element={<ProductPage />} />
          <Route path="seller" element={<SellerPage />} />
          <Route path="seller/:id" element={<SellerPage />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </main>
      <footer className="border-t border-line/70 py-6 text-center text-xs text-muted">
        Kala Studio · demo SellerOps. Tidak ada pembayaran sungguhan.
      </footer>
    </div>
  );
}
