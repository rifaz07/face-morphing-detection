"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { SignInButton, SignUpButton, UserButton, useUser } from "@clerk/nextjs";
import { ThemeToggle } from "@/components/shared/theme-toggle";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetTrigger, SheetClose } from "@/components/ui/sheet";
import { ScanFace, Menu, LayoutDashboard } from "lucide-react";
import { cn } from "@/lib/utils";
import { motion } from "framer-motion";

const NAV_LINKS = [
  { label: "Home", href: "/" },
  { label: "How It Works", href: "/how-it-works" },
  { label: "About", href: "/about" },
  { label: "Pricing", href: "/pricing" },
  { label: "Contact", href: "/contact" },
];

export function Navbar() {
  const pathname = usePathname();
  const { isSignedIn } = useUser();

  return (
    <header className="sticky top-0 z-50 w-full border-b border-border/50 bg-background/80 backdrop-blur-xl">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="flex h-16 items-center justify-between">

          {/* Logo */}
          <Link href="/" className="flex items-center gap-2.5 font-bold text-lg select-none" aria-label="FaceGuard home">
            <span className="gradient-brand rounded-lg p-1.5 shrink-0">
              <ScanFace className="size-5 text-white" aria-hidden />
            </span>
            <span className="hidden sm:block gradient-brand-text font-bold tracking-tight">
              FaceGuard
            </span>
          </Link>

          {/* Desktop nav */}
          <nav className="hidden md:flex items-center gap-1" aria-label="Main navigation">
            {NAV_LINKS.map(({ label, href }) => (
              <Link
                key={href}
                href={href}
                className={cn(
                  "relative px-3.5 py-2 rounded-md text-sm font-medium transition-colors",
                  "text-muted-foreground hover:text-foreground",
                  pathname === href && "text-foreground"
                )}
              >
                {label}
                {pathname === href && (
                  <motion.span
                    layoutId="nav-underline"
                    className="absolute inset-x-2 -bottom-px h-px gradient-brand rounded-full"
                  />
                )}
              </Link>
            ))}
          </nav>

          {/* Right controls */}
          <div className="flex items-center gap-1.5">
            <ThemeToggle />

            {/* Auth controls — desktop */}
            {!isSignedIn ? (
              <>
                <SignInButton mode="modal">
                  <Button variant="ghost" size="sm" className="hidden sm:flex text-sm">
                    Sign In
                  </Button>
                </SignInButton>
                <SignUpButton mode="modal">
                  <Button
                    size="sm"
                    className="hidden sm:flex gradient-brand text-white hover:opacity-90 transition-opacity text-sm font-medium"
                  >
                    Get Started
                  </Button>
                </SignUpButton>
              </>
            ) : (
              <>
                <Button variant="ghost" size="sm" className="hidden sm:flex gap-1.5 text-sm" asChild>
                  <Link href="/dashboard">
                    <LayoutDashboard className="size-4" />
                    Dashboard
                  </Link>
                </Button>
                <UserButton afterSignOutUrl="/" />
              </>
            )}

            {/* Mobile hamburger */}
            <Sheet>
              <SheetTrigger asChild>
                <Button variant="ghost" size="icon" className="md:hidden" aria-label="Open menu">
                  <Menu className="size-5" />
                </Button>
              </SheetTrigger>
              <SheetContent side="right" className="w-72 pt-10">
                <div className="flex flex-col gap-1 mt-4">
                  {NAV_LINKS.map(({ label, href }) => (
                    <SheetClose asChild key={href}>
                      <Link
                        href={href}
                        className={cn(
                          "px-4 py-2.5 rounded-lg text-sm font-medium transition-colors",
                          "text-muted-foreground hover:text-foreground hover:bg-muted",
                          pathname === href && "text-foreground bg-muted"
                        )}
                      >
                        {label}
                      </Link>
                    </SheetClose>
                  ))}
                  <div className="pt-3 mt-2 border-t flex flex-col gap-2">
                    {!isSignedIn ? (
                      <>
                        <SheetClose asChild>
                          <SignInButton mode="modal">
                            <Button variant="outline" size="sm" className="w-full">
                              Sign In
                            </Button>
                          </SignInButton>
                        </SheetClose>
                        <SheetClose asChild>
                          <SignUpButton mode="modal">
                            <Button size="sm" className="w-full gradient-brand text-white hover:opacity-90">
                              Get Started
                            </Button>
                          </SignUpButton>
                        </SheetClose>
                      </>
                    ) : (
                      <SheetClose asChild>
                        <Button variant="outline" size="sm" className="w-full justify-start gap-2" asChild>
                          <Link href="/dashboard">
                            <LayoutDashboard className="size-4" />
                            Dashboard
                          </Link>
                        </Button>
                      </SheetClose>
                    )}
                  </div>
                </div>
              </SheetContent>
            </Sheet>
          </div>
        </div>
      </div>
    </header>
  );
}
