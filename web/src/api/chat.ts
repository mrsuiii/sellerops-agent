import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../lib/api";
import { useSellerUser } from "../lib/seller-user";
import { sellerKeys } from "./seller";

export type ChatTurnMessage = { role: "user" | "agent"; text: string };
export type ChatTool = { name: string; ok: boolean };
export type ChatReply = { thread_id: string; reply: string; tools: ChatTool[]; proposals: string[] };

const chatUrl = (threadId: string, rest = "") => `/api/seller/chat/threads/${threadId}${rest}`;

export const chatKeys = {
  history: (threadId: string) => ["seller", "chat", threadId] as const,
};

export function useChatHistory(threadId: string, enabled: boolean) {
  return useQuery({
    queryKey: chatKeys.history(threadId),
    enabled,
    queryFn: async () => (await api<{ messages: ChatTurnMessage[] }>(chatUrl(threadId))).messages,
    staleTime: 0,
  });
}

/** One message to the seller agent. The reply can create caption proposals, so the seller queue is refreshed too. */
export function useSendChat(threadId: string) {
  const queryClient = useQueryClient();
  const { user } = useSellerUser();
  return useMutation({
    mutationFn: (text: string) =>
      api<ChatReply>(chatUrl(threadId, "/messages"), { method: "POST", body: { text }, seller: user }),
    onSettled: () => {
      void queryClient.invalidateQueries({ queryKey: chatKeys.history(threadId) });
      void queryClient.invalidateQueries({ queryKey: sellerKeys.root });
    },
  });
}
