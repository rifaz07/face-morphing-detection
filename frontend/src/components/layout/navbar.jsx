"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { ThemeToggle } from "@/components/shared/theme-toggle";
import { Button } from "@/components/ui/button";
import { ScanFace, Menu, X } from "lucide-react";
import { useState } from "react";
import { cn } from "@/lib/utils";

/** Public nav links shown before authentication. */
const NAV_LINKS = [
  { label: "How It Works", href: "/how-it-works" },
  { label: "About", href: "/about" },
  { label: "Pricing", href: "/pricing" },
  { label: "Contact", href: "/contact" },
];

/**
 * Top navigation bar — logo, links, theme toggle, and auth actions.
 * Mobile menu collapses into a hamburger toggle.
 */
export function Navbar() {
  const pathname = usePathname();
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <header className="sticky top-0 z-50 w-full border-b bg-background/80 backdrop-blur-sm">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="flex h-16 items-center justify-between">

          {/* Logo */}
          <Link
            href="/"
            className="flex items-center gap-2 font-bold text-lg select-none"
            aria-label="Face Morphing Detection — home"
          >
            <span className="gradient-brand rounded-lg p-1.5">
              <ScanFace className="size-5 text-white" aria-hidden />
            </span>
            <span className="hidden sm:block gradient-brand-text">
              FaceGuard
            </span>
          </Link>

          {/* Desktop nav links */}
          <nav className="hidden md:flex items-center gap-1" aria-label="Main navigation">
            {NAV_LINKS.map(({ label, href }) => (
              <Link
                key={href}
                href={href}
                className={cn(
                  "px-3 py-2 rounded-md text-sm font-medium transition-colors",
                  "text-muted-foreground hover:text-foreground hover:bg-muted",
                  pathname === href && "text-foreground bg-muted"
                )}
              >
                {label}
              </Link>
            ))}
          </nav>

          {/* Right-side controls */}
          <div className="flex items-center gap-2">
            <ThemeToggle />

            {/* Auth buttons — placeholder until Clerk is wired */}
            <Button variant="ghost" size="sm" className="hidden sm:flex" asChild>
              <Link href="/sign-in">Sign in</Link>
            </Button>
            <Button size="sm" className="hidden sm:flex gradient-brand text-white hover:opacity-90" asChild>
              <Link href="/sign-up">Get started</Link>
            </Button>

            {/* Mobile hamburger */}
            <button
              className="md:hidden p-2 rounded-md text-muted-foreground hover:text-foreground"
              onClick={() => setMobileOpen((o) => !o)}
              aria-label={mobileOpen ? "Close menu" : "Open menu"}
              aria-expanded={mobileOpen}
            >
              {mobileOpen ? <X className="size-5" /> : <Menu className="size-5" />}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile dropdown */}
      {mobileOpen && (
        <nav
          className="md:hidden border-t bg-background px-4 py-3 flex flex-col gap-1"
          aria-label="Mobile navigation"
        >
          {NAV_LINKS.map(({ label, href }) => (
            <Link
              key={href}
              href={href}
              className={cn(
                "px-3 py-2 rounded-md text-sm font-medium transition-colors",
                "text-muted-foreground hover:text-foreground hover:bg-muted",
                pathname === href && "text-foreground bg-muted"
              )}
              onClick={() => setMobileOpen(false)}
            >
              {label}
            </Link>
          ))}
          <div className="flex gap-2 pt-2 border-t mt-1">
            <Button variant="ghost" size="sm" className="flex-1" asChild>
              <Link href="/sign-in">Sign in</Link>
            </Button>
            <Button size="sm" className="flex-1 gradient-brand text-white hover:opacity-90" asChild>
              <Link href="/sign-up">Get started</Link>
            </Button>
          </div>
        </nav>
      )}
    </header>
  );
}
