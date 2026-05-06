import Link from "next/link";
import { ScanFace, Github } from "lucide-react";
import { Separator } from "@/components/ui/separator";

const FOOTER_LINKS = {
  Product: [
    { label: "How It Works", href: "/how-it-works" },
    { label: "Pricing", href: "/pricing" },
    { label: "Dashboard", href: "/dashboard" },
    { label: "Detect", href: "/detect" },
  ],
  Company: [
    { label: "About", href: "/about" },
    { label: "Contact", href: "/contact" },
    { label: "Privacy Policy", href: "/privacy" },
    { label: "Terms of Service", href: "/terms" },
  ],
  Resources: [
    { label: "GitHub", href: "https://github.com/rifaz07/face-morphing-detection" },
    { label: "API Docs", href: "/docs" },
    { label: "Changelog", href: "/changelog" },
  ],
};

/**
 * Site footer — logo, grouped links, copyright.
 * This is a Server Component (no 'use client' needed).
 */
export function Footer() {
  const year = new Date().getFullYear();

  return (
    <footer className="border-t bg-background" aria-label="Site footer">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-2 gap-8 sm:grid-cols-2 md:grid-cols-4">

          {/* Brand column */}
          <div className="col-span-2 sm:col-span-1">
            <Link
              href="/"
              className="flex items-center gap-2 font-bold text-lg"
              aria-label="FaceGuard home"
            >
              <span className="gradient-brand rounded-lg p-1.5">
                <ScanFace className="size-5 text-white" aria-hidden />
              </span>
              <span className="gradient-brand-text">FaceGuard</span>
            </Link>
            <p className="mt-3 text-sm text-muted-foreground max-w-xs">
              Detect morphed face images in real-time with our clustering-based
              ML pipeline. Protecting biometric systems from face morphing
              attacks.
            </p>
            <a
              href="https://github.com/rifaz07/face-morphing-detection"
              target="_blank"
              rel="noopener noreferrer"
              className="mt-4 inline-flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground transition-colors"
              aria-label="View source on GitHub"
            >
              <Github className="size-4" aria-hidden />
              View on GitHub
            </a>
          </div>

          {/* Link columns */}
          {Object.entries(FOOTER_LINKS).map(([section, links]) => (
            <div key={section}>
              <h3 className="text-sm font-semibold mb-3">{section}</h3>
              <ul className="space-y-2">
                {links.map(({ label, href }) => {
                  const isExternal = href.startsWith("http");
                  return (
                    <li key={label}>
                      <Link
                        href={href}
                        target={isExternal ? "_blank" : undefined}
                        rel={isExternal ? "noopener noreferrer" : undefined}
                        className="text-sm text-muted-foreground hover:text-foreground transition-colors"
                      >
                        {label}
                      </Link>
                    </li>
                  );
                })}
              </ul>
            </div>
          ))}
        </div>

        <Separator className="my-8" />

        <div className="flex flex-col sm:flex-row items-center justify-between gap-4 text-sm text-muted-foreground">
          <p>© {year} FaceGuard. Built for academic research.</p>
          <p>
            Made with ❤️ by{" "}
            <span className="font-medium text-foreground">rifaz shaikh razak</span>
          </p>
        </div>
      </div>
    </footer>
  );
}
