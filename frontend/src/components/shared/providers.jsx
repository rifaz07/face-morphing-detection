"use client";

import { ThemeProvider } from "next-themes";
import { Toaster } from "@/components/ui/sonner";
import { Analytics } from "@vercel/analytics/react";

/**
 * Client-side providers wrapper.
 * Groups all context providers that require the browser environment.
 * Placed here so layout.js stays a Server Component.
 */
export function Providers({ children }) {
  return (
    <ThemeProvider
      attribute="class"
      defaultTheme="system"
      enableSystem
      disableTransitionOnChange
    >
      {children}
      <Toaster richColors position="top-right" closeButton />
      <Analytics />
    </ThemeProvider>
  );
}
