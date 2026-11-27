// Shapes of the FastAPI responses the UI reads. Mirrors src/sellerops/store/api.py and
// src/sellerops/seller/api.py. The backend returns plain dicts, so these are written by hand.

export type Review = { rating: number; comment: string; reviewer: string; date: string };

export type ProductSummary = {
  product_id: string;
  name: string;
  brand: string;
  category: string;
  thumbnail: string;
  price_idr: number;
  final_price_idr: number;
  discount_percent: number;
  rating: number;
  stock: number;
};

export type Product = ProductSummary & {
  description: string;
  images: string[];
  reviews: Review[];
  sku?: string;
  warranty?: string;
  shipping?: string;
  return_policy?: string;
};

export type Recommendation = ProductSummary & { score: number; reasons: string[] };

export type Categories = { categories: { category: string; count: number }[]; brands: string[] };

export type ProductPage = { total: number; page: number; page_size: number; items: ProductSummary[] };

export type ProductQuery = {
  q: string;
  category: string;
  audience: string;
  brand: string;
  max_price: string;
  min_rating: string;
  in_stock: boolean;
  sort: string;
  page: number;
  page_size: number;
};

export type ChangeField = "name" | "description" | "tags";
export type FieldChange = { old: string | string[] | null; new: string | string[] | null };

export type Proposal = {
  proposal_id: string;
  product_id: string;
  status: "pending" | "applied" | "rejected" | "cancelled" | "stale" | "reverted";
  changes: Partial<Record<ChangeField, FieldChange>>;
  reason: string;
  created_by: string;
  created_at: string;
  decided_by?: string | null;
  feedback?: string | null;
  revision_of?: string | null;
  rejections_in_chain: number;
  reverted_by?: string | null;
  reverted_at?: string | null;
  inputs?: {
    images?: { sha256: string; source_url?: string; width: number; height: number }[];
    seller_facts?: Record<string, string | null>;
  };
};

export type CaptionRequest = {
  request_id: string;
  product_id: string;
  status: "open" | "proposed" | "cancelled";
  fields: ChangeField[];
  instruction?: string | null;
  created_by: string;
  created_at: string;
  proposal_id?: string | null;
};

export type ProposalListItem = Pick<Proposal, "proposal_id" | "product_id" | "status" | "changes" | "rejections_in_chain">;

export type Question = { key: string; question: string };

/** Approval graph state, as the workflow API reports it. */
export type ApprovalView = {
  state?: "running" | "done" | "error" | string;
  waiting_for?: "decision" | "answer" | null;
  stop_reason?: string | null;
  draft_error?: string | null;
  proposal_id?: string | null;
  request_id?: string | null;
  request?: {
    proposal?: { proposal_id: string };
    fields?: ChangeField[];
    questions?: Question[];
    error?: string | null;
  };
};

export type ApprovalResult = ApprovalView;

export type AuditEntry = {
  at: string;
  actor: string;
  action: string;
  proposal_id?: string | null;
  detail?: { request_id?: string };
};

export type Fact = { value: string | null; by: string; at: string; source?: string | null };

export type LlmBudget = {
  enabled: boolean;
  remaining_hour: number;
  per_client_hour: number;
};
