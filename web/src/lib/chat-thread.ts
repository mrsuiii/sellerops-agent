import { useCallback, useState } from "react";

// The browser keeps one conversation id; the server keeps the messages (per store).
const KEY = "kala-chat-thread";
const ID = /^[a-f0-9]{32}$/;

function newId(): string {
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}

function stored(): string | null {
  try {
    const value = localStorage.getItem(KEY);
    return value && ID.test(value) ? value : null;
  } catch {
    return null;
  }
}

export function useChatThread(): { threadId: string; startNew: () => void } {
  const [threadId, setThreadId] = useState<string>(() => {
    const id = stored() ?? newId();
    rememberThread(id);
    return id;
  });
  const startNew = useCallback(() => {
    const id = newId();
    rememberThread(id);
    setThreadId(id);
  }, []);
  return { threadId, startNew };
}

function rememberThread(id: string): void {
  try {
    localStorage.setItem(KEY, id);
  } catch {
    /* ignore */
  }
}
