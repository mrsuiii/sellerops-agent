import { useCallback } from "react";
import { useNavigate } from "react-router";
import { useToast } from "../../components/toast";
import type { ApprovalResult } from "../../api/types";

/** What the agent stopped on, in words. `done` with one of these reasons shows as a notice on the proposal. */
export const STOP_REASON: Record<string, [tone: "ok" | "bad" | "neutral", text: string]> = {
  applied: ["ok", "Disetujui. Caption baru sudah tampil di toko."],
  stale: ["bad", "Tidak diterapkan: caption produk sudah berubah lebih dulu, jadi usulan ini basi."],
  cancelled: ["neutral", "Dibatalkan."],
  max_rejections: ["bad", "Sudah ditolak 5 kali, alur berhenti. Buat usulan baru bila masih perlu."],
  revision_failed: ["bad", "Agent dua kali gagal menulis revisi yang valid, alur berhenti."],
};

/**
 * After the agent answers a write, move the seller to whatever it is waiting for next:
 * a new proposal to decide on, or a question on the request. Otherwise show a toast.
 */
export function useThreadResult(currentId: string) {
  const navigate = useNavigate();
  const toast = useToast();
  return useCallback(
    (result: ApprovalResult & { request_id?: string | null; proposal_id?: string | null }, doneText?: string) => {
      const next = result.waiting_for === "decision" ? result.request?.proposal?.proposal_id ?? null : null;
      if (next && next !== currentId) {
        toast.show(`${currentId.startsWith("CR-") ? "Usulan" : "Revisi"} ${next} siap diputuskan`);
        navigate(`/seller/${next}`);
        return;
      }
      if (result.waiting_for === "answer" && result.request_id && currentId !== result.request_id && !result.proposal_id) {
        navigate(`/seller/${result.request_id}`);
        return;
      }
      if (doneText) toast.show(doneText);
    },
    [currentId, navigate, toast],
  );
}
