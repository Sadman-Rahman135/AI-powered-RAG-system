import type { Metadata } from "next";
import type { ReactNode } from "react";

import { AppHeader } from "@/components/app-header";
import { Providers } from "@/components/providers";

import { plexMono, publicSans } from "./fonts";
import "./globals.css";

export const metadata: Metadata = {
  title: { default: "DocIntel", template: "%s – DocIntel" },
  description: "Invoice and document intelligence: upload, extract, validate, review, search and compare.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className={`${publicSans.variable} ${plexMono.variable} h-full antialiased`}>
      <body className="flex min-h-full flex-col">
        <a
          href="#main"
          className="sr-only rounded-md bg-primary px-3 py-2 text-primary-foreground focus:not-sr-only focus:absolute focus:top-2 focus:left-2 focus:z-50"
        >
          Skip to content
        </a>
        <Providers>
          <AppHeader />
          <main id="main" tabIndex={-1} className="mx-auto flex w-full max-w-[1440px] flex-1 flex-col gap-6 px-4 py-6 outline-none sm:px-6">
            {children}
          </main>
        </Providers>
      </body>
    </html>
  );
}
