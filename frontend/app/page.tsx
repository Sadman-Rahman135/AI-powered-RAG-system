import type { Metadata } from "next";

import { DashboardView } from "@/components/dashboard-view";

export const metadata: Metadata = { title: "Documents" };

export default function DashboardPage() {
  return <DashboardView />;
}
