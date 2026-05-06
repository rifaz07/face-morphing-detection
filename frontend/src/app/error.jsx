"use client";

import { useEffect } from "react";
import { motion } from "framer-motion";
import { RefreshCw, Home, AlertTriangle } from "lucide-react";
import { Button } from "@/components/ui/button";
import Link from "next/link";

export default function Error({ error, reset }) {
  useEffect(() => {
    console.error(error);
  }, [error]);

  const code = error?.digest ?? "UNKNOWN";

  return (
    <main className="min-h-[calc(100vh-4rem)] flex items-center justify-center px-4">
      <div className="text-center max-w-md mx-auto">
        {/* Icon */}
        <motion.div
          initial={{ opacity: 0, scale: 0.7 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
          className="flex justify-center mb-6"
        >
          <div className="w-20 h-20 rounded-2xl gradient-brand flex items-center justify-center shadow-lg">
            <AlertTriangle className="size-10 text-white" />
          </div>
        </motion.div>

        {/* Heading */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.15 }}
          className="space-y-3"
        >
          <h1 className="text-3xl font-bold">Something went wrong</h1>
          <p className="text-muted-foreground leading-relaxed">
            We hit an unexpected error on our end. Our team has been notified.
            You can try again or head back to the home page.
          </p>
        </motion.div>

        {/* Error code */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.3 }}
          className="mt-5 inline-flex items-center gap-2 rounded-lg border bg-muted/50 px-3 py-1.5"
        >
          <span className="text-xs text-muted-foreground font-mono">Error code:</span>
          <span className="text-xs font-mono font-semibold">{code}</span>
        </motion.div>

        {/* Actions */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.4 }}
          className="mt-8 flex flex-col sm:flex-row items-center justify-center gap-3"
        >
          <Button
            onClick={reset}
            className="gradient-brand text-white hover:opacity-90 gap-2 h-11 px-6"
          >
            <RefreshCw className="size-4" /> Try Again
          </Button>
          <Button asChild variant="outline" className="gap-2 h-11 px-6">
            <Link href="/">
              <Home className="size-4" /> Back to Home
            </Link>
          </Button>
        </motion.div>
      </div>
    </main>
  );
}
