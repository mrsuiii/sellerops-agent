import { describe, expect, it } from "vitest";
import { rupiah, whoLabel } from "./format";

describe("format", () => {
  it("formats rupiah with dot separators", () => {
    expect(rupiah(1250000)).toBe("Rp1.250.000");
  });

  it("names the agent and strips the seller prefix", () => {
    expect(whoLabel("agent:caption-approval")).toBe("agent (revisi)");
    expect(whoLabel("seller:dewi")).toBe("dewi");
  });
});
