import { expect, test } from "@playwright/test";

test("halaman tidak ditemukan tampil 404", async ({ page }) => {
  await page.goto("/tidak-ada");
  await expect(page.getByText("Halaman tidak ditemukan")).toBeVisible();
});
