import { expect, test } from "@playwright/test";

// Serial: the tests share one demo database. CP-0001 and CP-0002 are the seeded pending proposals.
test.describe.configure({ mode: "serial" });

test("kuota habis tampil sebagai toast, tanpa mengubah usulan", async ({ page }) => {
  await page.route("**/api/seller/caption-proposals/CP-0001/approval/decision", (route) =>
    route.fulfill({
      status: 429,
      contentType: "application/json",
      body: JSON.stringify({
        detail: "Batas demo tercapai: 10 langkah agent per jam. Setujui dan batalkan tetap gratis.",
        code: "llm_budget_exhausted",
      }),
    }),
  );
  await page.goto("/seller/CP-0001");
  await page.getByRole("button", { name: "Tolak…" }).click();
  await page.getByRole("textbox", { name: /Apa yang perlu diubah/ }).fill("lebih pendek");
  await page.getByRole("button", { name: "Tolak & minta revisi" }).click();

  await expect(page.getByRole("alert").filter({ hasText: "Batas demo tercapai" })).toBeVisible();
  await expect(page).toHaveURL(/\/seller\/CP-0001$/);
});

test("tolak dengan alasan membuat revisi baru", async ({ page }) => {
  await page.goto("/seller/CP-0001");
  await page.getByRole("button", { name: "Tolak…" }).click();
  await page.getByRole("textbox", { name: /Apa yang perlu diubah/ }).fill("lebih pendek");
  await page.getByRole("button", { name: "Tolak & minta revisi" }).click();

  await expect(page).not.toHaveURL(/CP-0001$/);
  await expect(page.getByText("(fake revision 1)")).toBeVisible();
});

test("setujui menerapkan caption dan tampil di toko", async ({ page }) => {
  await page.goto("/seller/CP-0002");
  await page.getByRole("button", { name: "Setujui & terapkan" }).click();

  await expect(page.getByText("Disetujui. Caption baru sudah tampil di toko.")).toBeVisible();
  await page.goto("/p/FS083");
  await expect(page.getByRole("heading", { level: 1, name: "Men's Blue & Black Check Shirt" })).toBeVisible();
});

test("caption baru: agent menulis usulan dari permintaan", async ({ page }) => {
  await page.goto("/seller");
  await page.getByRole("button", { name: "Caption baru" }).click();
  await page.getByRole("dialog", { name: "Caption baru" }).getByRole("button", { name: "Minta agent menulis" }).click();

  await expect(page).toHaveURL(/\/seller\/CP-\d+$/);
  await expect(page.getByText("(fake revision")).toBeVisible();
});

test("pertanyaan agent: jawaban terkirim sebagai answers", async ({ page }) => {
  // The fake model never asks, so the question state is served by a stub; the form and its payload are real.
  await page.route("**/api/seller/caption-requests/CR-0001", (route) =>
    route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        request_id: "CR-0001",
        product_id: "FS083",
        status: "open",
        fields: ["description"],
        instruction: "",
        created_by: "seller-demo",
        created_at: "2026-11-27T10:00:00+00:00",
        proposal_id: null,
      }),
    }),
  );
  await page.route("**/api/seller/caption-requests/CR-0001/approval", (route) =>
    route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        state: "running",
        waiting_for: "answer",
        request_id: "CR-0001",
        request: { questions: [{ key: "material", question: "Bahan apa?" }], error: null },
      }),
    }),
  );
  let posted: unknown = null;
  await page.route("**/api/seller/caption-requests/CR-0001/approval/answer", async (route) => {
    posted = route.request().postDataJSON();
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ state: "running" }) });
  });

  await page.goto("/seller/CR-0001");
  await expect(page.getByText("Agent butuh info dulu sebelum menulis caption")).toBeVisible();
  await page.getByLabel(/Bahan apa/).fill("katun");
  await page.getByRole("button", { name: "Kirim jawaban" }).click();

  await expect.poll(() => posted).toEqual({ answers: { material: "katun" } });
});
