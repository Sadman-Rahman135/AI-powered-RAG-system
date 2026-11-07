/**
 * TanStack Query hooks for every API resource. Lists and details poll every 2 s while anything they show is
 * queued or processing, and stop once it settles. Mutations update or invalidate the caches they affect.
 */
import { keepPreviousData, useMutation, useQuery, useQueryClient, type QueryClient } from "@tanstack/react-query";

import {
  api,
  type CrossVerificationListParams,
  type DocumentListParams,
  type InvoiceListParams,
  type SearchApiParams,
} from "@/lib/api";
import { POLL_INTERVAL_MS, isActiveDocument, refetchWhileActive } from "@/lib/status";
import type { CrossVerifyRequest, Document, DocumentDetail, Invoice, InvoicePatch, Paged, Stats } from "@/lib/types";

export const queryKeys = {
  stats: () => ["stats"] as const,
  config: () => ["config"] as const,
  documents: () => ["documents"] as const,
  documentLists: () => ["documents", "list"] as const,
  documentList: (params: DocumentListParams) => ["documents", "list", params] as const,
  document: (id: string) => ["documents", "detail", id] as const,
  documentPages: (id: string, offset: number, limit: number, version: string) =>
    ["documents", "detail", id, "pages", { offset, limit, version }] as const,
  documentEvents: (id: string) => ["documents", "detail", id, "events"] as const,
  invoices: () => ["invoices"] as const,
  invoiceList: (params: InvoiceListParams) => ["invoices", "list", params] as const,
  invoice: (id: string) => ["invoices", "detail", id] as const,
  corrections: (invoiceId: string) => ["invoices", "detail", invoiceId, "corrections"] as const,
  searches: () => ["search"] as const,
  search: (params: SearchApiParams) => ["search", params] as const,
  allCrossVerifications: () => ["cross-verifications"] as const,
  crossVerifications: (params: CrossVerificationListParams) => ["cross-verifications", params] as const,
  picker: (kind: string, query: string) => ["picker", kind, query] as const,
};

const documentsInFlight = refetchWhileActive<Paged<Document>>((data) => data?.items ?? []);

function statsInFlight(data: Stats | undefined): number | false {
  const byStatus = data?.documents.by_status;
  return (byStatus?.queued ?? 0) + (byStatus?.processing ?? 0) > 0 ? POLL_INTERVAL_MS : false;
}

export function useStats() {
  return useQuery({
    queryKey: queryKeys.stats(),
    queryFn: ({ signal }) => api.getStats(signal),
    refetchInterval: (query) => statsInFlight(query.state.data),
  });
}

export function useConfig() {
  return useQuery({
    queryKey: queryKeys.config(),
    queryFn: ({ signal }) => api.getConfig(signal),
    staleTime: Infinity,
    retry: false,
  });
}

export function useDocuments(params: DocumentListParams) {
  return useQuery({
    queryKey: queryKeys.documentList(params),
    queryFn: ({ signal }) => api.listDocuments(params, signal),
    placeholderData: keepPreviousData,
    refetchInterval: (query) => documentsInFlight(query.state.data),
  });
}

export function useDocument(id: string) {
  return useQuery({
    queryKey: queryKeys.document(id),
    queryFn: ({ signal }) => api.getDocument(id, signal),
    refetchInterval: (query) => (query.state.data && isActiveDocument(query.state.data) ? POLL_INTERVAL_MS : false),
  });
}

/** `version` (the document's `updated_at`) makes page text refetch after (re)processing. */
export function useDocumentPages(id: string, offset: number, limit: number, version: string) {
  return useQuery({
    queryKey: queryKeys.documentPages(id, offset, limit, version),
    queryFn: ({ signal }) => api.getDocumentPages(id, { offset, limit }, signal),
    placeholderData: keepPreviousData,
  });
}

export function useDocumentEvents(id: string, active: boolean) {
  return useQuery({
    queryKey: queryKeys.documentEvents(id),
    queryFn: ({ signal }) => api.getDocumentEvents(id, signal),
    refetchInterval: active ? POLL_INTERVAL_MS : false,
  });
}

export function useInvoices(params: InvoiceListParams) {
  return useQuery({
    queryKey: queryKeys.invoiceList(params),
    queryFn: ({ signal }) => api.listInvoices(params, signal),
    placeholderData: keepPreviousData,
  });
}

export function useCorrections(invoiceId: string | null) {
  return useQuery({
    queryKey: queryKeys.corrections(invoiceId ?? ""),
    queryFn: ({ signal }) => api.getCorrections(invoiceId as string, signal),
    enabled: invoiceId !== null,
  });
}

export function useSearch(params: SearchApiParams, enabled: boolean) {
  return useQuery({
    queryKey: queryKeys.search(params),
    queryFn: ({ signal }) => api.search(params, signal),
    enabled,
    placeholderData: keepPreviousData,
  });
}

export function useCrossVerifications(params: CrossVerificationListParams, version: string) {
  return useQuery({
    queryKey: [...queryKeys.crossVerifications(params), version],
    queryFn: ({ signal }) => api.listCrossVerifications(params, signal),
  });
}

/**
 * Refetch document lists, details and stats after the server's documents changed (upload, reprocess, delete).
 * A fetch still in flight started before the change, so it is cancelled first: TanStack Query would otherwise reuse
 * it for a query with no data yet (the page's first load), and the table would miss a just-uploaded document until
 * something else refetched it.
 */
export async function refreshAfterDocumentChange(queryClient: QueryClient): Promise<void> {
  const keys = [queryKeys.documents(), queryKeys.stats()];
  await Promise.all(keys.map((queryKey) => queryClient.cancelQueries({ queryKey })));
  await Promise.all(keys.map((queryKey) => queryClient.invalidateQueries({ queryKey })));
}

export function useReprocessDocument() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => api.reprocessDocument(id),
    onSuccess: () => void refreshAfterDocumentChange(queryClient),
  });
}

export function useDeleteDocument() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => api.deleteDocument(id),
    onSuccess: (_data, id) => {
      queryClient.removeQueries({ queryKey: queryKeys.document(id) });
      void refreshAfterDocumentChange(queryClient);
      void queryClient.invalidateQueries({ queryKey: queryKeys.invoices() });
    },
  });
}

export function useReprocessFailed() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () => api.reprocessFailed(),
    onSuccess: () => void refreshAfterDocumentChange(queryClient),
  });
}

/** Put the server's updated record into the document cache at once, then refetch what re-validation changed. */
function applyInvoiceUpdate(queryClient: QueryClient, updated: Invoice) {
  queryClient.setQueryData<DocumentDetail>(queryKeys.document(updated.document_id), (previous) =>
    previous
      ? { ...previous, invoices: previous.invoices.map((invoice) => (invoice.id === updated.id ? updated : invoice)) }
      : previous,
  );
  queryClient.setQueryData(queryKeys.invoice(updated.id), updated);
  void queryClient.invalidateQueries({ queryKey: queryKeys.document(updated.document_id) });
  void queryClient.invalidateQueries({ queryKey: queryKeys.corrections(updated.id) });
  void queryClient.invalidateQueries({ queryKey: queryKeys.invoices() });
  void queryClient.invalidateQueries({ queryKey: queryKeys.documentLists() });
  void queryClient.invalidateQueries({ queryKey: queryKeys.stats() });
  void queryClient.invalidateQueries({ queryKey: queryKeys.searches() });
  void queryClient.invalidateQueries({ queryKey: queryKeys.allCrossVerifications() });
}

export function usePatchInvoice() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ id, patch }: { id: string; patch: InvoicePatch }) => api.patchInvoice(id, patch),
    onSuccess: (updated) => applyInvoiceUpdate(queryClient, updated),
  });
}

export function useApproveInvoice() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ id, comment }: { id: string; comment: string | null }) => api.approveInvoice(id, comment),
    onSuccess: (updated) => applyInvoiceUpdate(queryClient, updated),
  });
}

export function useCrossVerify() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (request: CrossVerifyRequest) => api.crossVerify(request),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: queryKeys.allCrossVerifications() });
    },
  });
}
