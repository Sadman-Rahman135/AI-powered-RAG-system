"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Suspense } from "react";

import { Badge } from "@/components/ui/badge";
import { formatCount } from "@/lib/format";
import { useStats } from "@/lib/queries";
import { cn } from "@/lib/utils";

const NAV = [
  { href: "/", label: "Documents" },
  { href: "/review", label: "Review queue" },
  { href: "/search", label: "Search" },
  { href: "/compare", label: "Compare" },
] as const;

function isActive(href: string, pathname: string | null): boolean {
  if (!pathname) return false;
  if (href === "/") return pathname === "/" || pathname.startsWith("/documents");
  return pathname === href || pathname.startsWith(`${href}/`);
}

function ReviewCount() {
  const { data } = useStats();
  const count = data?.invoices.by_review_status.needs_review ?? 0;
  if (count === 0) return null;
  return (
    <Badge variant="warning" className="figures">
      {formatCount(count)}
      <span className="sr-only"> records to review</span>
    </Badge>
  );
}

function NavLinks({ pathname }: { pathname: string | null }) {
  return (
    <ul className="flex items-center gap-1">
      {NAV.map((item) => {
        const active = isActive(item.href, pathname);
        return (
          <li key={item.href}>
            <Link
              href={item.href}
              aria-current={active ? "page" : undefined}
              className={cn(
                "flex h-14 items-center gap-2 border-b-2 border-transparent px-3 text-sm font-medium text-muted-foreground transition-colors hover:text-foreground",
                active && "border-primary text-foreground",
              )}
            >
              {item.label}
              {item.href === "/review" ? <ReviewCount /> : null}
            </Link>
          </li>
        );
      })}
    </ul>
  );
}

function ActiveNavLinks() {
  return <NavLinks pathname={usePathname()} />;
}

function Mark() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" className="size-6 shrink-0">
      <rect x="1.5" y="1.5" width="21" height="21" rx="4" className="fill-primary" />
      <path d="M6 8h8M6 12h5M6 16h4" stroke="white" strokeWidth="1.6" strokeLinecap="round" />
      <path d="m13.5 15.5 2.2 2.2L19 13" fill="none" stroke="white" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export function AppHeader() {
  return (
    <header className="sticky top-0 z-30 border-b bg-card">
      <div className="mx-auto flex h-14 w-full max-w-[1440px] items-center gap-6 px-4 sm:px-6">
        <Link href="/" className="flex items-center gap-2 rounded-sm font-semibold tracking-tight">
          <Mark />
          <span>DocIntel</span>
        </Link>
        <nav aria-label="Main" className="min-w-0 overflow-x-auto">
          {/* The active link depends on the URL, which is unknown while a dynamic route is prerendered. */}
          <Suspense fallback={<NavLinks pathname={null} />}>
            <ActiveNavLinks />
          </Suspense>
        </nav>
      </div>
    </header>
  );
}
