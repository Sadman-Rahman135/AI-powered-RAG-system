"use client";

/** Confirmation before reprocessing a document: reprocessing replaces its extracted records. */
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";

export interface ReprocessDialogProps {
  filename: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onConfirm: () => void;
}

export function ReprocessDialog({ filename, open, onOpenChange, onConfirm }: ReprocessDialogProps) {
  return (
    <AlertDialog open={open} onOpenChange={onOpenChange}>
      <AlertDialogContent data-testid="reprocess-dialog">
        <AlertDialogHeader>
          <AlertDialogTitle>Reprocess {filename}?</AlertDialogTitle>
          <AlertDialogDescription>
            Re-processing replaces the extracted records. Corrections and approvals stay in the audit history but are
            not re-applied.
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>Cancel</AlertDialogCancel>
          <AlertDialogAction
            onClick={() => {
              onOpenChange(false);
              onConfirm();
            }}
            data-testid="confirm-reprocess"
          >
            Reprocess document
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
