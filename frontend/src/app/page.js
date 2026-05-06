import { ThemeToggle } from "@/components/shared/theme-toggle";

/**
 * Temporary landing placeholder.
 * Replace this file when building the real landing page (feature/landing-page).
 */
export default function HomePage() {
  return (
    <main className="min-h-screen flex flex-col items-center justify-center gap-6 px-4 text-center">
      {/* Theme toggle — top-right corner */}
      <div className="fixed top-4 right-4">
        <ThemeToggle />
      </div>

      {/* Badge */}
      <span className="inline-flex items-center rounded-full border px-3 py-1 text-xs font-medium text-muted-foreground">
        Final Year Project · 2025
      </span>

      {/* Heading */}
      <h1 className="text-4xl sm:text-6xl font-bold tracking-tight max-w-3xl gradient-brand-text">
        Face Morphing Detection System
      </h1>

      <p className="text-muted-foreground text-lg max-w-xl">
        A clustering-based ML pipeline that detects morphed face images in
        real-time using LBP + DCT features and K-Means classification.
      </p>

      {/* Under construction notice */}
      <div className="glass rounded-xl px-6 py-4 mt-4 max-w-sm">
        <p className="text-sm text-muted-foreground">
          🚧 &nbsp;Full landing page coming soon — scaffold is ready.
        </p>
      </div>
    </main>
  );
}
