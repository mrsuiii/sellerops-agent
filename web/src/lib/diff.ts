export type Segment = { text: string; kind: "same" | "added" | "removed" };

/**
 * Word-level diff with an LCS table. Whitespace is kept as its own tokens so the
 * rendered text looks the same as the source; only words get marked.
 * `old` returns the segments for the left side, `new` for the right side.
 */
export function wordDiff(oldText: string, newText: string): { old: Segment[]; new: Segment[] } {
  const a = tokenize(oldText);
  const b = tokenize(newText);
  const n = a.length;
  const m = b.length;
  const lcs = Array.from({ length: n + 1 }, () => new Uint32Array(m + 1));
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      lcs[i]![j] = a[i] === b[j] ? lcs[i + 1]![j + 1]! + 1 : Math.max(lcs[i + 1]![j]!, lcs[i]![j + 1]!);
    }
  }
  const left: Segment[] = [];
  const right: Segment[] = [];
  let i = 0;
  let j = 0;
  while (i < n || j < m) {
    if (i < n && j < m && a[i] === b[j]) {
      left.push({ text: a[i]!, kind: "same" });
      right.push({ text: b[j]!, kind: "same" });
      i++;
      j++;
    } else if (j < m && (i >= n || lcs[i]![j + 1]! >= lcs[i + 1]![j]!)) {
      right.push({ text: b[j]!, kind: isSpace(b[j]!) ? "same" : "added" });
      j++;
    } else {
      left.push({ text: a[i]!, kind: isSpace(a[i]!) ? "same" : "removed" });
      i++;
    }
  }
  return { old: left, new: right };
}

export function tokenize(text: string): string[] {
  return text.split(/(\s+)/).filter((token) => token.length > 0);
}

const isSpace = (token: string): boolean => /^\s+$/.test(token);
