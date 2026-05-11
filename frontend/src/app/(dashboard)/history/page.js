"use client";

import { useEffect, useState } from "react";
import Image from "next/image";
import Link from "next/link";
import { ScanFace, ChevronDown, ChevronUp } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";

const FILTERS = ["All", "Real", "Morphed"];

function ResultBadge({ result }) {
  if (result === "REAL")
    return (
      <Badge className="bg-green-500/15 text-green-600 dark:text-green-400 border-green-500/20 shrink-0">
        Real
      </Badge>
    );
  if (result === "MORPHED")
    return (
      <Badge className="bg-red-500/15 text-red-600 dark:text-red-400 border-red-500/20 shrink-0">
        Morphed
      </Badge>
    );
  return (
    <Badge variant="secondary" className="shrink-0">
      {result}
    </Badge>
  );
}

function PredictionRow({ p }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="divide-y divide-border/30">
      <button
        type="button"
        className="w-full text-left grid md:grid-cols-[56px_1fr_auto_auto_auto_auto] items-center gap-4 px-5 py-3.5 hover:bg-muted/40 transition-colors"
        onClick={() => setExpanded((v) => !v)}
        aria-expanded={expanded}
      >
        {/* Thumbnail */}
        <div className="size-10 rounded-lg overflow-hidden bg-muted flex items-center justify-center shrink-0">
          {p.imageUrl ? (
            <Image
              src={p.imageUrl}
              alt={p.imageName}
              width={40}
              height={40}
              className="object-cover size-10"
              unoptimized={false}
            />
          ) : (
            <ScanFace className="size-5 text-muted-foreground" />
          )}
        </div>

        {/* Name */}
        <span className="text-sm font-medium truncate pr-2">{p.imageName}</span>

        {/* Result */}
        <ResultBadge result={p.result} />

        {/* Confidence */}
        <span className="text-sm text-muted-foreground hidden md:block w-16 text-right">
          {p.confidence != null ? `${(p.confidence * 100).toFixed(1)}%` : "—"}
        </span>

        {/* Date */}
        <span className="text-sm text-muted-foreground hidden md:block whitespace-nowrap">
          {new Date(p.createdAt).toLocaleDateString("en-GB", {
            day: "2-digit",
            month: "short",
            year: "numeric",
          })}
        </span>

        {/* Expand toggle */}
        <span className="text-muted-foreground ml-1 shrink-0">
          {expanded ? (
            <ChevronUp className="size-4" />
          ) : (
            <ChevronDown className="size-4" />
          )}
        </span>
      </button>

      {expanded && (
        <div className="px-5 py-4 bg-muted/20 grid sm:grid-cols-2 gap-4 text-sm">
          {p.imageUrl && (
            <div className="sm:col-span-2 flex items-start gap-4">
              <Image
                src={p.imageUrl}
                alt={p.imageName}
                width={120}
                height={120}
                className="rounded-xl object-cover border border-border/50"
                unoptimized={false}
              />
              <div className="space-y-1 text-xs text-muted-foreground">
                <p>
                  <span className="font-medium text-foreground">File:</span> {p.imageName}
                </p>
                <p>
                  <span className="font-medium text-foreground">Size:</span>{" "}
                  {p.imageSize ? `${(p.imageSize / 1024).toFixed(1)} KB` : "—"}
                </p>
                {p.metadata?.width && (
                  <p>
                    <span className="font-medium text-foreground">Dimensions:</span>{" "}
                    {p.metadata.width}×{p.metadata.height}px
                  </p>
                )}
                <p>
                  <span className="font-medium text-foreground">Uploaded:</span>{" "}
                  {new Date(p.createdAt).toLocaleString()}
                </p>
              </div>
            </div>
          )}
          <div>
            <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-2">
              Detection Result
            </p>
            <dl className="space-y-1 text-xs">
              <div className="flex justify-between">
                <dt className="text-muted-foreground">Verdict</dt>
                <dd className="font-medium">{p.result}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-muted-foreground">Confidence</dt>
                <dd className="font-medium">
                  {p.confidence != null ? `${(p.confidence * 100).toFixed(2)}%` : "—"}
                </dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-muted-foreground">Faces found</dt>
                <dd className="font-medium">{p.faceCount ?? "—"}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-muted-foreground">Processing time</dt>
                <dd className="font-medium">
                  {p.processingTimeMs != null ? `${p.processingTimeMs.toFixed(1)}ms` : "—"}
                </dd>
              </div>
            </dl>
          </div>
        </div>
      )}
    </div>
  );
}

export default function HistoryPage() {
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState("All");

  useEffect(() => {
    fetch("/api/predictions")
      .then((r) => r.json())
      .then((d) => setPredictions(d.predictions ?? []))
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  const filtered = predictions.filter((p) => {
    if (filter === "Real") return p.result === "REAL";
    if (filter === "Morphed") return p.result === "MORPHED";
    return true;
  });

  return (
    <div className="p-6 md:p-8 max-w-5xl mx-auto space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold">Detection History</h1>
          <p className="text-muted-foreground mt-1">All your past image analyses.</p>
        </div>

        {/* Filter buttons */}
        <div className="flex gap-1.5">
          {FILTERS.map((f) => (
            <Button
              key={f}
              size="sm"
              variant={filter === f ? "default" : "outline"}
              className={cn(
                "text-xs",
                filter === f && "gradient-brand text-white hover:opacity-90 border-transparent"
              )}
              onClick={() => setFilter(f)}
            >
              {f}
            </Button>
          ))}
        </div>
      </div>

      <Card className="border-border/50 overflow-hidden">
        <CardHeader className="py-3 px-5 border-b border-border/50">
          <CardTitle className="text-xs text-muted-foreground font-normal uppercase tracking-wide">
            {loading
              ? "Loading…"
              : `${filtered.length} detection${filtered.length !== 1 ? "s" : ""}${filter !== "All" ? ` · ${filter}` : ""}`}
          </CardTitle>
        </CardHeader>

        {loading ? (
          <div className="divide-y divide-border/50">
            {Array.from({ length: 4 }).map((_, i) => (
              <div key={i} className="flex items-center gap-4 px-5 py-4">
                <Skeleton className="size-10 rounded-lg shrink-0" />
                <div className="flex-1 space-y-2">
                  <Skeleton className="h-4 w-48" />
                  <Skeleton className="h-3 w-24" />
                </div>
                <Skeleton className="h-5 w-16 rounded-full" />
              </div>
            ))}
          </div>
        ) : filtered.length === 0 ? (
          <CardContent className="py-16 flex flex-col items-center gap-3 text-center">
            <ScanFace className="size-10 text-muted-foreground/40" aria-hidden />
            <p className="text-muted-foreground">
              {filter !== "All"
                ? `No ${filter.toLowerCase()} detections yet.`
                : "No detections yet."}
            </p>
            {filter === "All" && (
              <Button size="sm" className="gradient-brand text-white hover:opacity-90 mt-1" asChild>
                <Link href="/detect">Start detecting</Link>
              </Button>
            )}
          </CardContent>
        ) : (
          <>
            {/* Table header */}
            <div className="hidden md:grid grid-cols-[56px_1fr_auto_auto_auto_auto] gap-4 px-5 py-2.5 border-b border-border/50 text-xs font-medium text-muted-foreground uppercase tracking-wide">
              <span>Image</span>
              <span>File name</span>
              <span>Result</span>
              <span className="w-16 text-right">Confidence</span>
              <span>Date</span>
              <span />
            </div>
            <div>
              {filtered.map((p) => (
                <PredictionRow key={p.id} p={p} />
              ))}
            </div>
          </>
        )}
      </Card>
    </div>
  );
}
