import { FileQuestion } from "lucide-react";
import Link from "next/link";

import { EmptyState } from "@/components/query-state";
import { buttonVariants } from "@/components/ui/button";

export default function NotFound() {
  return (
    <EmptyState icon={FileQuestion} title="This page does not exist" description="Check the address, or start again from the document list.">
      <Link href="/" className={buttonVariants({ variant: "outline" })}>
        Back to documents
      </Link>
    </EmptyState>
  );
}
