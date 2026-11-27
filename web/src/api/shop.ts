import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";
import type { Categories, Product, ProductPage, ProductQuery, Recommendation } from "./types";

export const shopKeys = {
  categories: ["shop", "categories"] as const,
  products: (query: ProductQuery) => ["shop", "products", query] as const,
  product: (id: string) => ["shop", "product", id] as const,
  similar: (id: string) => ["shop", "similar", id] as const,
};

export function useCategories() {
  return useQuery({
    queryKey: shopKeys.categories,
    queryFn: () => api<Categories>("/api/categories"),
    staleTime: 5 * 60_000,
  });
}

export function useProducts(query: ProductQuery) {
  return useQuery({
    queryKey: shopKeys.products(query),
    queryFn: () => {
      const params = new URLSearchParams({ page: String(query.page), page_size: String(query.page_size), sort: query.sort });
      const add = (key: string, value: string) => value && params.set(key, value);
      add("q", query.q);
      add("category", query.category);
      add("audience", query.audience);
      add("brand", query.brand);
      add("max_price", query.max_price);
      add("min_rating", query.min_rating);
      if (query.in_stock) params.set("in_stock", "true");
      return api<ProductPage>(`/api/products?${params}`);
    },
    placeholderData: keepPreviousData,
  });
}

export function useProduct(id: string) {
  return useQuery({
    queryKey: shopKeys.product(id),
    queryFn: () => api<Product>(`/api/products/${encodeURIComponent(id)}`),
    retry: (count, error) => !(error instanceof Error && "status" in error && error.status === 404) && count < 2,
  });
}

export function useSimilar(id: string) {
  return useQuery({
    queryKey: shopKeys.similar(id),
    queryFn: async () => {
      const { items } = await api<{ items: Recommendation[] }>("/api/recommend", {
        method: "POST",
        body: { similar_to: id, limit: 4 },
      });
      return items;
    },
  });
}
