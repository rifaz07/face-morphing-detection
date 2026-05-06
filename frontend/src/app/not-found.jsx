"use client";

import { motion } from "framer-motion";
import { Home, ArrowLeft } from "lucide-react";
import { Button } from "@/components/ui/button";
import Link from "next/link";

const SHAPES = [
  { size: 180, x: "10%", y: "15%", delay: 0 },
  { size: 120, x: "80%", y: "10%", delay: 0.4 },
  { size: 90,  x: "70%", y: "70%", delay: 0.8 },
  { size: 60,  x: "15%", y: "75%", delay: 1.2 },
];

export default function NotFound() {
  return (
    <main className="relative min-h-[calc(100vh-4rem)] flex items-center justify-center overflow-hidden px-4">
      {/* Floating ambient shapes */}
      {SHAPES.map(({ size, x, y, delay }, i) => (
        <motion.div
          key={i}
          className="absolute rounded-full gradient-brand opacity-[0.06] pointer-events-none"
          style={{ width: size, height: size, left: x, top: y }}
          animate={{ y: [0, -20, 0], scale: [1, 1.05, 1] }}
          transition={{ duration: 5 + i, repeat: Infinity, delay, ease: "easeInOut" }}
        />
      ))}

      <div className="relative z-10 text-center max-w-lg mx-auto">
        {/* 404 number */}
        <motion.div
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
        >
          <span
            className="gradient-brand-text text-[clamp(7rem,25vw,12rem)] font-black leading-none select-none"
            aria-hidden="true"
          >
            404
          </span>
        </motion.div>

        {/* Message */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.55, delay: 0.2 }}
          className="mt-4 space-y-3"
        >
          <h1 className="text-2xl sm:text-3xl font-bold">Page not found</h1>
          <p className="text-muted-foreground text-base sm:text-lg leading-relaxed">
            Looks like this page got morphed into nothing&nbsp;👻
          </p>
          <p className="text-sm text-muted-foreground">
            The URL may have changed, or the page may have been removed.
          </p>
        </motion.div>

        {/* Actions */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.55, delay: 0.35 }}
          className="mt-8 flex flex-col sm:flex-row items-center justify-center gap-3"
        >
          <Button asChild className="gradient-brand text-white hover:opacity-90 gap-2 h-11 px-6">
            <Link href="/">
              <Home className="size-4" /> Back to Home
            </Link>
          </Button>
          <Button
            variant="outline"
            className="gap-2 h-11 px-6"
            onClick={() => window.history.back()}
          >
            <ArrowLeft className="size-4" /> Go Back
          </Button>
        </motion.div>
      </div>
    </main>
  );
}
