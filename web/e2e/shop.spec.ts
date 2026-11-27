import { expect, test } from "@playwright/test";

test.describe("toko", () => {
  test("katalog memuat dan pencarian menulis q ke URL", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByText("54 produk")).toBeVisible();

    await page.getByRole("searchbox", { name: "Cari produk" }).fill("dress");
    await page.getByRole("button", { name: "Cari" }).click();

    await expect(page).toHaveURL(/q=dress/);
    await expect(page.getByText(/produk untuk/)).toBeVisible();
  });

  test("filter kategori dan halaman detail", async ({ page }) => {
    await page.goto("/?category=watches");
    await expect(page.getByRole("button", { name: /Jam tangan/ })).toHaveAttribute("aria-pressed", "true");
    await page.getByRole("link", { name: /Rolex/ }).first().click();

    await expect(page).toHaveURL(/\/p\/FS/);
    await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
    await expect(page.getByRole("heading", { name: "Mungkin kamu suka" })).toBeVisible();
  });

  test("tambah ke keranjang memperbarui penghitung", async ({ page }) => {
    await page.goto("/p/FS173");
    await page.getByRole("button", { name: "Tambah ke keranjang" }).click();

    await expect(page.getByRole("status").filter({ hasText: "masuk keranjang" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Keranjang, 1 barang" })).toBeVisible();
  });
});
