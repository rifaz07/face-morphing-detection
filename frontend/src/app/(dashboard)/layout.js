"use client";

import { Sidebar } from "@/components/layout/sidebar";
import { useSyncUser } from "@/hooks/use-sync-user";

export default function DashboardLayout({ children }) {
  useSyncUser();

  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <main className="flex-1 overflow-y-auto bg-muted/20">
        {children}
      </main>
    </div>
  );
}
