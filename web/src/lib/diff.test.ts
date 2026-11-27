import { describe, expect, it } from "vitest";
import { tokenize, wordDiff } from "./diff";

const text = (segments: { text: string }[]) => segments.map((s) => s.text).join("");

describe("wordDiff", () => {
  it("keeps both sides intact", () => {
    const { old, new: next } = wordDiff("Kemeja katun putih", "Kemeja katun putih lembut");
    expect(text(old)).toBe("Kemeja katun putih");
    expect(text(next)).toBe("Kemeja katun putih lembut");
  });

  it("marks only the changed words", () => {
    const { old, new: next } = wordDiff("Tas kulit cokelat", "Tas kulit hitam");
    expect(old.filter((s) => s.kind === "removed").map((s) => s.text)).toEqual(["cokelat"]);
    expect(next.filter((s) => s.kind === "added").map((s) => s.text)).toEqual(["hitam"]);
  });

  it("returns no marks for identical text", () => {
    const { old, new: next } = wordDiff("sama", "sama");
    expect([...old, ...next].every((s) => s.kind === "same")).toBe(true);
  });

  it("handles an empty side", () => {
    const { new: next } = wordDiff("", "baru");
    expect(next).toEqual([{ text: "baru", kind: "added" }]);
  });
});

describe("tokenize", () => {
  it("keeps whitespace tokens so the text can be rebuilt", () => {
    expect(tokenize("a  b").join("")).toBe("a  b");
  });
});
