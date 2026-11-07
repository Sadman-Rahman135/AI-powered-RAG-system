"use client";

import { useQueryClient } from "@tanstack/react-query";
import { RotateCw } from "lucide-react";

import { DocumentsTable } from "@/components/documents-table";
import { PageHeader } from "@/components/page-header";
import { StatsStrip } from "@/components/stats-strip";
import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import { toast } from "@/components/ui/toast";
import { UploadDropzone } from "@/components/upload-dropzone";
import { errorMessage } from "@/lib/api";
import { formatCount } from "@/lib/format";
import { refreshAfterDocumentChange, useConfig, useReprocessFailed, useStats } from "@/lib/queries";

function ReprocessFailedButton() {
  const { data } = useStats();
  const reprocess = useReprocessFailed();
  const failed = data?.documents.by_status.failed ?? 0;
  if (failed === 0) return null;

  return (
    <Button
      type="button"
      variant="outline"
      disabled={reprocess.isPending}
      data-testid="reprocess-failed"
      onClick={() =>
        reprocess.mutate(undefined, {
          onSuccess: (result) =>
            toast.add({
              title: result.requeued === 1 ? "1 document queued again" : `${formatCount(result.requeued)} documents queued again`,
              type: "success",
            }),
          onError: (error) => toast.add({ title: "Could not reprocess", description: errorMessage(error), type: "error" }),
        })
      }
    >
      {reprocess.isPending ? <Spinner data-icon="inline-start" /> : <RotateCw aria-hidden="true" data-icon="inline-start" />}
      Reprocess {formatCount(failed)} failed
    </Button>
  );
}

export function DashboardView() {
  const queryClient = useQueryClient();
  const { data: config } = useConfig();

  return (
    <>
      <PageHeader
        title="Documents"
        description="Upload invoices and receipts, follow their processing, and open anything that needs a second look."
        actions={<ReprocessFailedButton />}
      />
      <StatsStrip />
      <UploadDropzone
        maxFilesPerRequest={config?.max_files_per_request}
        maxUploadMb={config?.max_upload_mb}
        maxRequestMb={config?.max_request_mb}
        allowedExtensions={config?.allowed_extensions?.length ? config.allowed_extensions : undefined}
        onUploaded={() => void refreshAfterDocumentChange(queryClient)}
      />
      <DocumentsTable />
    </>
  );
}
