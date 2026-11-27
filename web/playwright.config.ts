import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig, devices } from "@playwright/test";

// End-to-end tests run against the real FastAPI store (demo mode, fake caption model)
// on its own port with a fresh database per run. Chrome is the one installed on the
// machine (channel "chrome"), so no browser download is needed.
const port = Number(process.env.E2E_PORT ?? 8001);
const data = mkdtempSync(join(tmpdir(), "kala-e2e-"));
const repoRoot = resolve(fileURLToPath(new URL(".", import.meta.url)), "..");

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: [["list"]],
  use: {
    baseURL: `http://127.0.0.1:${port}`,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  projects: [
    {
      name: "chrome",
      use: { ...devices["Desktop Chrome"], channel: "chrome" },
    },
  ],
  webServer: {
    command: `uv run uvicorn sellerops.store.api:app --host 127.0.0.1 --port ${port}`,
    cwd: repoRoot,
    url: `http://127.0.0.1:${port}/api/health`,
    reuseExistingServer: false,
    stdout: "ignore",
    timeout: 60_000,
    env: {
      SELLEROPS_DEMO_MODE: "1",
      SELLEROPS_APPROVAL_FAKE: "1",
      SELLEROPS_DEMO_LLM_PER_CLIENT_HOUR: "100",
      SELLEROPS_SELLER_DB: join(data, "seller.sqlite"),
      SELLEROPS_APPROVALS_DB: join(data, "approvals.sqlite"),
      SELLEROPS_LOG_FORMAT: "text",
    },
  },
});
