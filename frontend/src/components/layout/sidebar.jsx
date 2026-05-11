"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { UserButton } from "@clerk/nextjs";
import { ScanFace, LayoutDashboard, Shield, History, BarChart2, Settings } from "lucide-react";
import { cn } from "@/lib/utils";

const NAV_ITEMS = [
  { label: "Dashboard", href: "/dashboard", icon: LayoutDashboard },
  { label: "Detect", href: "/detect", icon: Shield },
  { label: "History", href: "/history", icon: History },
  { label: "Analytics", href: "/analytics", icon: BarChart2 },
  { label: "Settings", href: "/settings", icon: Settings },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="hidden md:flex flex-col w-60 shrink-0 h-screen sticky top-0 border-r border-border/50 bg-background/95 backdrop-blur-sm">
      {/* Logo */}
      <div className="flex items-center gap-2.5 px-5 h-16 border-b border-border/50">
        <span className="gradient-brand rounded-lg p-1.5 shrink-0">
          <ScanFace className="size-5 text-white" aria-hidden />
        </span>
        <span className="gradient-brand-text font-bold text-lg tracking-tight">FaceGuard</span>
      </div>

      {/* Navigation */}
      <nav className="flex-1 overflow-y-auto px-3 py-4 space-y-0.5" aria-label="Sidebar navigation">
        {NAV_ITEMS.map(({ label, href, icon: Icon }) => {
          const isActive = pathname === href || pathname.startsWith(`${href}/`);
          return (
            <Link
              key={href}
              href={href}
              className={cn(
                "flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all",
                isActive
                  ? "bg-purple-500/10 text-purple-600 dark:text-purple-400"
                  : "text-muted-foreground hover:text-foreground hover:bg-muted"
              )}
              aria-current={isActive ? "page" : undefined}
            >
              <Icon
                className={cn(
                  "size-4 shrink-0",
                  isActive ? "text-purple-600 dark:text-purple-400" : "text-muted-foreground"
                )}
                aria-hidden
              />
              {label}
              {isActive && (
                <span className="ml-auto h-1.5 w-1.5 rounded-full bg-purple-600 dark:bg-purple-400" />
              )}
            </Link>
          );
        })}
      </nav>

      {/* User section */}
      <div className="px-4 py-4 border-t border-border/50">
        <div className="flex items-center gap-3">
          <UserButton afterSignOutUrl="/" />
          <span className="text-xs text-muted-foreground truncate">My Account</span>
        </div>
      </div>
    </aside>
  );
}
