import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ApiError, api } from "../lib/api";
import { useSellerUser } from "../lib/seller-user";
import type {
  ApprovalResult,
  ApprovalView,
  AuditEntry,
  CaptionRequest,
  ChangeField,
  Fact,
  LlmBudget,
  Proposal,
  ProposalListItem,
  ProductPage,
} from "./types";

export const sellerKeys = {
  root: ["seller"] as const,
  proposals: (status: string) => ["seller", "proposals", status] as const,
  requests: (status: string) => ["seller", "requests", status] as const,
  proposal: (id: string) => ["seller", "proposal", id] as const,
  proposalApproval: (id: string) => ["seller", "proposal-approval", id] as const,
  request: (id: string) => ["seller", "request", id] as const,
  requestApproval: (id: string) => ["seller", "request-approval", id] as const,
  audit: (productId: string) => ["seller", "audit", productId] as const,
  facts: (productId: string) => ["seller", "facts", productId] as const,
  budget: ["seller", "budget"] as const,
  products: ["seller", "products"] as const,
};

const proposalUrl = (id: string, rest = "") => `/api/seller/caption-proposals/${encodeURIComponent(id)}${rest}`;
const requestUrl = (id: string, rest = "") => `/api/seller/caption-requests/${encodeURIComponent(id)}${rest}`;

/** The workflow thread is keyed by the request (CR-…) or the proposal (CP-…/…). */
export const threadUrl = (id: string, rest = "") => (id.startsWith("CR-") ? requestUrl(id, rest) : proposalUrl(id, rest));

/** Approval is only there once the graph has started; "not found" means nothing to show yet. */
async function approvalOrNull(path: string): Promise<ApprovalView | null> {
  try {
    return await api<ApprovalView>(path);
  } catch (error) {
    if (error instanceof ApiError && error.code === "approval_not_found") return null;
    throw error;
  }
}

// ---------- queries ----------

export function useProposalQueue(status: string) {
  return useQuery({
    queryKey: sellerKeys.proposals(status),
    queryFn: async () => {
      const params = new URLSearchParams({ limit: "100" });
      if (status) params.set("status", status);
      return (await api<{ proposals: ProposalListItem[] }>(`/api/seller/caption-proposals?${params}`)).proposals;
    },
  });
}

export function useRequestQueue(status: string) {
  return useQuery({
    queryKey: sellerKeys.requests(status),
    queryFn: async () => {
      const params = new URLSearchParams({ limit: "50" });
      if (status) params.set("status", "open"); // "Menunggu": requests the agent is still on
      return (await api<{ requests: CaptionRequest[] }>(`/api/seller/caption-requests?${params}`)).requests;
    },
  });
}

export function useProposal(id: string) {
  return useQuery({ queryKey: sellerKeys.proposal(id), queryFn: () => api<Proposal>(proposalUrl(id)) });
}

export function useProposalApproval(id: string, enabled = true) {
  return useQuery({
    queryKey: sellerKeys.proposalApproval(id),
    queryFn: () => approvalOrNull(proposalUrl(id, "/approval")),
    enabled,
  });
}

export function useCaptionRequest(id: string) {
  return useQuery({ queryKey: sellerKeys.request(id), queryFn: () => api<CaptionRequest>(requestUrl(id)) });
}

export function useRequestApproval(id: string, enabled = true) {
  return useQuery({
    queryKey: sellerKeys.requestApproval(id),
    queryFn: () => approvalOrNull(requestUrl(id, "/approval")),
    enabled,
  });
}

export function useAuditLog(productId: string | undefined) {
  return useQuery({
    queryKey: sellerKeys.audit(productId ?? ""),
    enabled: Boolean(productId),
    queryFn: async () =>
      (await api<{ entries: AuditEntry[] }>(`/api/seller/audit-log?product_id=${encodeURIComponent(productId ?? "")}&limit=30`)).entries,
  });
}

export function useProductFacts(productId: string | undefined) {
  return useQuery({
    queryKey: sellerKeys.facts(productId ?? ""),
    enabled: Boolean(productId),
    queryFn: async () =>
      (await api<{ facts: Record<string, Fact> }>(`/api/seller/products/${encodeURIComponent(productId ?? "")}/facts`)).facts,
  });
}

export function useLlmBudget() {
  return useQuery({
    queryKey: sellerKeys.budget,
    queryFn: () => api<LlmBudget>("/api/seller/llm-budget"),
    staleTime: 30_000,
  });
}

/** Product names and thumbnails for the seller lists (includes approved captions). */
export function useSellerProducts() {
  return useQuery({
    queryKey: sellerKeys.products,
    queryFn: () => api<ProductPage>("/api/products?page_size=60"),
    select: (page) => Object.fromEntries(page.items.map((item) => [item.product_id, item])),
  });
}

// ---------- mutations ----------

/** Every write goes through here: it sends the seller header and refreshes what the write touched. */
function useSellerWrite<TVars, TResult>(send: (vars: TVars, seller: string) => Promise<TResult>) {
  const queryClient = useQueryClient();
  const { user } = useSellerUser();
  return useMutation({
    mutationFn: (vars: TVars) => send(vars, user),
    onSettled: () => {
      // Approve/revert change the shop's captions too, so refresh every seller and shop list.
      void queryClient.invalidateQueries({ queryKey: sellerKeys.root });
      void queryClient.invalidateQueries({ queryKey: ["shop"] });
    },
  });
}

export type Decision = { decision: "approve" | "reject" | "cancel"; feedback?: string; add_fields?: ChangeField[] };

export function useDecide(proposalId: string) {
  return useSellerWrite<Decision, ApprovalResult>((body, seller) =>
    api<ApprovalResult>(proposalUrl(proposalId, "/approval/decision"), { method: "POST", body, seller }));
}

export function useAnswer(threadId: string) {
  return useSellerWrite<Record<string, string | null>, ApprovalResult>((answers, seller) =>
    api<ApprovalResult>(threadUrl(threadId, "/approval/answer"), { method: "POST", body: { answers }, seller }));
}

export function useRetry(threadId: string) {
  return useSellerWrite<void, ApprovalResult>((_, seller) =>
    api<ApprovalResult>(threadUrl(threadId, "/approval/retry"), { method: "POST", seller }));
}

export function useRevert(proposalId: string) {
  return useSellerWrite<void, Proposal>((_, seller) =>
    api<Proposal>(proposalUrl(proposalId, "/revert"), { method: "POST", body: {}, seller }));
}

export function useCancelRequest(requestId: string) {
  return useSellerWrite<void, CaptionRequest>((_, seller) =>
    api<CaptionRequest>(requestUrl(requestId, "/cancel"), { method: "POST", seller }));
}

export type RequestCreate = { product_id: string; fields: ChangeField[]; instruction: string };
export type ProposalCreate = {
  product_id: string;
  reason: string;
  name?: string;
  description?: string;
  tags?: string[];
};

/** Result of starting a caption request: the agent may have a question, a draft, or failed. */
export type RequestStart = ApprovalResult & { request_id?: string; proposal_id?: string | null; state?: string };

export function useCreateRequest() {
  return useSellerWrite<RequestCreate, RequestStart>((body, seller) =>
    api<RequestStart>("/api/seller/caption-requests", { method: "POST", body: { ...body }, seller }));
}

export function useCreateProposal() {
  return useSellerWrite<ProposalCreate, Proposal>((body, seller) =>
    api<Proposal>("/api/seller/caption-proposals", { method: "POST", body: { ...body }, seller }));
}
