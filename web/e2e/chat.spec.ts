import { expect, test } from "@playwright/test";

// The e2e server runs the fake chat model (SELLEROPS_APPROVAL_FAKE=1): it echoes the question.
test("chat dengan agent: kirim pesan, lihat balasan, riwayat tetap setelah reload", async ({ page }) => {
  await page.goto("/seller");
  await page.getByRole("button", { name: "Chat agent" }).click();

  const dialog = page.getByRole("dialog", { name: "Chat dengan agent" });
  await dialog.getByRole("textbox", { name: "Pesan untuk agent" }).fill("berapa stok FS180?");
  await dialog.getByRole("textbox", { name: "Pesan untuk agent" }).press("Enter");

  await expect(dialog.getByText("(fake agent)")).toBeVisible();
  await expect(dialog.getByText("berapa stok FS180?").first()).toBeVisible();

  await page.reload();
  await page.getByRole("button", { name: "Chat agent" }).click();
  await expect(page.getByRole("dialog", { name: "Chat dengan agent" }).getByText("(fake agent)")).toBeVisible();
});

test("chat kosong tidak mengirim pesan", async ({ page }) => {
  await page.goto("/seller");
  await page.getByRole("button", { name: "Chat agent" }).click();
  const dialog = page.getByRole("dialog", { name: "Chat dengan agent" });
  await expect(dialog.getByRole("button", { name: "Kirim" })).toBeDisabled();
});
