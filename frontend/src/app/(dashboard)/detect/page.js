"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import { useDropzone } from "react-dropzone";
import Link from "next/link";
import { motion, AnimatePresence } from "framer-motion";
import {
  Upload,
  ScanFace,
  X,
  CheckCircle2,
  AlertCircle,
  Loader2,
  Check,
  History,
  RotateCcw,
  Clock,
  Users,
  Maximize2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { toast } from "sonner";

const MAX_SIZE_MB = 10;

const PIPELINE_STEPS = [
  "Uploading image to cloud",
  "Detecting face with Haar Cascade",
  "Extracting LBP + DCT features",
  "Classifying with K-Means",
];

function ProgressSteps({ activeStep, done }) {
  return (
    <div className="space-y-3">
      {PIPELINE_STEPS.map((label, i) => {
        const isComplete = done || i < activeStep;
        const isActive = !done && i === activeStep;
        return (
          <motion.div
            key={label}
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: i * 0.1 }}
            className="flex items-center gap-3"
          >
            <div
              className={cn(
                "size-6 rounded-full flex items-center justify-center shrink-0 transition-all",
                isComplete
                  ? "bg-green-500"
                  : isActive
                  ? "bg-purple-500"
                  : "bg-muted border border-border"
              )}
            >
              {isComplete ? (
                <Check className="size-3.5 text-white" />
              ) : isActive ? (
                <Loader2 className="size-3.5 text-white animate-spin" />
              ) : (
                <span className="text-[10px] text-muted-foreground font-medium">{i + 1}</span>
              )}
            </div>
            <span
              className={cn(
                "text-sm transition-colors",
                isComplete
                  ? "text-green-600 dark:text-green-400 font-medium"
                  : isActive
                  ? "text-foreground font-medium"
                  : "text-muted-foreground"
              )}
            >
              {label}
            </span>
          </motion.div>
        );
      })}
    </div>
  );
}

function ConfidenceCounter({ target }) {
  const [display, setDisplay] = useState(0);

  useEffect(() => {
    const pct = Math.round(target * 100);
    let current = 0;
    const step = Math.max(1, Math.floor(pct / 50));
    const timer = setInterval(() => {
      current = Math.min(current + step, pct);
      setDisplay(current);
      if (current >= pct) clearInterval(timer);
    }, 20);
    return () => clearInterval(timer);
  }, [target]);

  return <span>{display}</span>;
}

export default function DetectPage() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [status, setStatus] = useState("idle"); // idle | loading | done | error
  const [activeStep, setActiveStep] = useState(0);
  const [result, setResult] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);
  const stepTimerRef = useRef(null);

  const onDrop = useCallback((accepted, rejected) => {
    if (rejected.length > 0) {
      toast.error("File rejected — must be JPG/PNG/WebP under 10 MB");
      return;
    }
    const f = accepted[0];
    if (!f) return;
    if (preview) URL.revokeObjectURL(preview);
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
    setErrorMsg(null);
    setStatus("idle");
  }, [preview]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/*": [".jpg", ".jpeg", ".png", ".webp"] },
    maxSize: MAX_SIZE_MB * 1024 * 1024,
    multiple: false,
  });

  function clearFile() {
    if (preview) URL.revokeObjectURL(preview);
    setFile(null);
    setPreview(null);
    setResult(null);
    setErrorMsg(null);
    setStatus("idle");
    setActiveStep(0);
    if (stepTimerRef.current) clearInterval(stepTimerRef.current);
  }

  async function handleAnalyze() {
    if (!file) return;
    setStatus("loading");
    setActiveStep(0);
    setResult(null);
    setErrorMsg(null);

    // Advance through steps 0→1→2 while waiting for the API
    let step = 0;
    stepTimerRef.current = setInterval(() => {
      step += 1;
      if (step < 3) {
        setActiveStep(step);
      } else {
        clearInterval(stepTimerRef.current);
      }
    }, 900);

    try {
      const form = new FormData();
      form.append("file", file);

      const res = await fetch("/api/detect", { method: "POST", body: form });
      clearInterval(stepTimerRef.current);

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        const msg =
          body.error === "NO_FACE"
            ? "No face detected. Please upload a clear, front-facing photo."
            : body.error === "INVALID_IMAGE"
            ? "Invalid image format. Try a different photo."
            : body.error === "SERVICE_UNAVAILABLE"
            ? "Detection service is offline. Make sure the backend is running."
            : body.message || "Something went wrong. Please try again.";
        setErrorMsg(msg);
        setStatus("error");
        return;
      }

      // Jump to final step before showing result
      setActiveStep(3);
      await new Promise((r) => setTimeout(r, 400));

      const data = await res.json();
      setResult(data);
      setStatus("done");
    } catch {
      clearInterval(stepTimerRef.current);
      setErrorMsg("Network error — check your connection and try again.");
      setStatus("error");
    }
  }

  return (
    <div className="p-6 md:p-8 max-w-2xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Detect Morphed Face</h1>
        <p className="text-muted-foreground mt-1">
          Upload a face image to run the full LBP + DCT + K-Means pipeline.
        </p>
      </div>

      {/* ── Drop zone (no file selected) ── */}
      {!file && (
        <Card
          {...getRootProps()}
          className={cn(
            "border-2 border-dashed cursor-pointer transition-all select-none",
            isDragActive
              ? "border-purple-500 bg-purple-500/5 scale-[1.01]"
              : "border-border/60 hover:border-purple-400/60 hover:bg-muted/30"
          )}
        >
          <CardContent className="py-16 flex flex-col items-center gap-4 text-center">
            <input {...getInputProps()} />
            <div className="rounded-full bg-purple-500/10 p-5">
              <Upload className="size-8 text-purple-500" aria-hidden />
            </div>
            <div>
              <p className="font-semibold text-lg">
                {isDragActive ? "Drop image here…" : "Drop your face image here"}
              </p>
              <p className="text-sm text-muted-foreground mt-1">
                or{" "}
                <span className="text-purple-500 underline underline-offset-2">
                  click to browse
                </span>
              </p>
            </div>
            <p className="text-xs text-muted-foreground">
              Supported: JPEG, PNG, WebP · Max {MAX_SIZE_MB} MB
            </p>
          </CardContent>
        </Card>
      )}

      {/* ── Preview + action ── */}
      {file && status !== "done" && (
        <Card className="border-border/50">
          <CardContent className="p-5 space-y-5">
            <div className="flex items-start gap-4">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={preview}
                alt="Selected face"
                className="size-24 object-cover rounded-xl border border-border/50 shrink-0"
              />
              <div className="flex-1 min-w-0">
                <p className="font-medium truncate">{file.name}</p>
                <p className="text-xs text-muted-foreground mt-0.5">
                  {(file.size / 1024).toFixed(1)} KB
                </p>
                {status === "idle" && (
                  <Button
                    variant="ghost"
                    size="sm"
                    className="text-muted-foreground hover:text-destructive gap-1.5 -ml-2 mt-2"
                    onClick={clearFile}
                  >
                    <X className="size-3.5" />
                    Change image
                  </Button>
                )}
              </div>
            </div>

            {/* Loading steps */}
            <AnimatePresence>
              {status === "loading" && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                  className="pt-2 border-t border-border/50"
                >
                  <p className="text-xs text-muted-foreground font-medium uppercase tracking-wide mb-4">
                    Processing pipeline
                  </p>
                  <ProgressSteps activeStep={activeStep} done={false} />
                </motion.div>
              )}
            </AnimatePresence>

            {/* Error */}
            {status === "error" && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="flex items-start gap-3 rounded-lg bg-destructive/10 border border-destructive/20 p-4"
              >
                <AlertCircle className="size-5 text-destructive shrink-0 mt-0.5" />
                <div className="flex-1">
                  <p className="text-sm font-medium text-destructive">{errorMsg}</p>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="mt-2 -ml-2 text-destructive hover:text-destructive/80 gap-1.5"
                    onClick={() => { setStatus("idle"); setErrorMsg(null); }}
                  >
                    <RotateCcw className="size-3.5" />
                    Try again
                  </Button>
                </div>
              </motion.div>
            )}

            {status === "idle" && (
              <Button
                className="w-full gradient-brand text-white hover:opacity-90 gap-2 h-11 text-base font-semibold"
                onClick={handleAnalyze}
              >
                <ScanFace className="size-5" />
                Analyze Image
              </Button>
            )}

            {status === "loading" && (
              <Button disabled className="w-full h-11 text-base font-semibold gap-2">
                <Loader2 className="size-5 animate-spin" />
                Analyzing…
              </Button>
            )}
          </CardContent>
        </Card>
      )}

      {/* ── Result card ── */}
      <AnimatePresence>
        {status === "done" && result && (
          <motion.div
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ type: "spring", stiffness: 200, damping: 20 }}
          >
            <Card
              className={cn(
                "border-2 overflow-hidden",
                result.prediction === "REAL"
                  ? "border-green-500/30"
                  : "border-red-500/30"
              )}
            >
              {/* Colour bar */}
              <div
                className={cn(
                  "h-1.5 w-full",
                  result.prediction === "REAL"
                    ? "bg-gradient-to-r from-green-400 to-emerald-500"
                    : "bg-gradient-to-r from-red-400 to-rose-500"
                )}
              />

              <CardContent className="pt-8 pb-6 px-6 space-y-6">
                {/* Main verdict */}
                <div className="flex flex-col items-center gap-3 text-center">
                  {result.prediction === "REAL" ? (
                    <CheckCircle2 className="size-16 text-green-500" />
                  ) : (
                    <AlertCircle className="size-16 text-red-500" />
                  )}
                  <div>
                    <p className="text-2xl font-bold">
                      {result.prediction === "REAL"
                        ? "Authentic Face Detected"
                        : "Morphed Face Detected"}
                    </p>
                    <p className="text-sm text-muted-foreground mt-1">
                      {result.prediction === "REAL"
                        ? "This image appears to be a genuine face photo."
                        : "Warning: This image shows signs of morphing attacks."}
                    </p>
                  </div>

                  {/* Confidence count-up */}
                  <div
                    className={cn(
                      "rounded-2xl px-6 py-4 text-center mt-2",
                      result.prediction === "REAL"
                        ? "bg-green-500/10"
                        : "bg-red-500/10"
                    )}
                  >
                    <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-1">
                      Confidence
                    </p>
                    <p
                      className={cn(
                        "text-5xl font-bold tabular-nums",
                        result.prediction === "REAL"
                          ? "text-green-600 dark:text-green-400"
                          : "text-red-600 dark:text-red-400"
                      )}
                    >
                      <ConfidenceCounter target={result.confidence} />
                      <span className="text-2xl ml-1">%</span>
                    </p>
                  </div>
                </div>

                {/* Details */}
                <div className="grid grid-cols-3 gap-3 pt-2 border-t border-border/50">
                  <div className="flex flex-col items-center gap-1 text-center">
                    <Users className="size-4 text-muted-foreground" />
                    <p className="text-lg font-semibold">{result.faceCount ?? 1}</p>
                    <p className="text-xs text-muted-foreground">Face{result.faceCount !== 1 ? "s" : ""} found</p>
                  </div>
                  <div className="flex flex-col items-center gap-1 text-center">
                    <Clock className="size-4 text-muted-foreground" />
                    <p className="text-lg font-semibold">
                      {result.processingTimeMs < 1000
                        ? `${Math.round(result.processingTimeMs)}ms`
                        : `${(result.processingTimeMs / 1000).toFixed(1)}s`}
                    </p>
                    <p className="text-xs text-muted-foreground">Processing time</p>
                  </div>
                  <div className="flex flex-col items-center gap-1 text-center">
                    <Maximize2 className="size-4 text-muted-foreground" />
                    <p className="text-lg font-semibold">
                      {result.imageDimensions
                        ? `${result.imageDimensions.width}×${result.imageDimensions.height}`
                        : "—"}
                    </p>
                    <p className="text-xs text-muted-foreground">Dimensions</p>
                  </div>
                </div>

                {/* Completed pipeline steps */}
                <div className="pt-2 border-t border-border/50">
                  <p className="text-xs text-muted-foreground font-medium uppercase tracking-wide mb-3">
                    Pipeline completed
                  </p>
                  <ProgressSteps activeStep={4} done={true} />
                </div>

                {/* Actions */}
                <div className="flex gap-3 pt-2">
                  <Button variant="outline" className="flex-1 gap-2" onClick={clearFile}>
                    <ScanFace className="size-4" />
                    Analyze Another
                  </Button>
                  <Button className="flex-1 gap-2 gradient-brand text-white hover:opacity-90" asChild>
                    <Link href="/history">
                      <History className="size-4" />
                      View in History
                    </Link>
                  </Button>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
