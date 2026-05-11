"use client";

import { useEffect, useState } from "react";
import { ScanFace } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";

function ResultBadge({ result }) {
  if (result === "REAL")
    return (
      <Badge className="bg-green-500/15 text-green-600 dark:text-green-400 border-green-500/20">
        Real
      </Badge>
    );
  if (result === "MORPHED")
    return (
      <Badge className="bg-red-500/15 text-red-600 dark:text-red-400 border-red-500/20">
        Morphed
      </Badge>
    );
  return <Badge variant="secondary">{result}</Badge>;
}

export default function HistoryPage() {
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/api/predictions")
      .then((r) => r.json())
      .then((d) => setPredictions(d.predictions ?? []))
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="p-6 md:p-8 max-w-5xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Detection History</h1>
        <p className="text-muted-foreground mt-1">All your past image analyses.</p>
      </div>

      <Card className="border-border/50">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm text-muted-foreground font-normal">
            {loading ? "Loading…" : `${predictions.length} total detection${predictions.length !== 1 ? "s" : ""}`}
          </CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          {loading ? (
            <div className="space-y-0 divide-y divide-border/50">
              {Array.from({ length: 5 }).map((_, i) => (
                <div key={i} className="px-6 py-4 flex items-center gap-4">
                  <Skeleton className="size-10 rounded-lg shrink-0" />
                  <div className="flex-1 space-y-2">
                    <Skeleton className="h-4 w-48" />
                    <Skeleton className="h-3 w-24" />
                  </div>
                  <Skeleton className="h-5 w-16 rounded-full" />
                </div>
              ))}
            </div>
          ) : predictions.length === 0 ? (
            <div className="py-16 flex flex-col items-center gap-3 text-center">
              <ScanFace className="size-10 text-muted-foreground/40" aria-hidden />
              <p className="text-muted-foreground">No detections yet.</p>
            </div>
          ) : (
            <>
              {/* Table header */}
              <div className="hidden md:grid grid-cols-[1fr_auto_auto_auto] gap-4 px-6 py-3 border-b border-border/50 text-xs font-medium text-muted-foreground uppercase tracking-wide">
                <span>Image</span>
                <span>Result</span>
                <span>Confidence</span>
                <span>Date</span>
              </div>
              <div className="divide-y divide-border/50">
                {predictions.map((p) => (
                  <div
                    key={p.id}
                    className="grid md:grid-cols-[1fr_auto_auto_auto] grid-cols-[1fr_auto] items-center gap-4 px-6 py-4"
                  >
                    <div className="flex items-center gap-3 min-w-0">
                      <div className="size-9 rounded-lg bg-muted flex items-center justify-center shrink-0">
                        <ScanFace className="size-4 text-muted-foreground" />
                      </div>
                      <span className="text-sm font-medium truncate">{p.imageName}</span>
                    </div>
                    <ResultBadge result={p.result} />
                    <span className="text-sm text-muted-foreground hidden md:block">
                      {p.confidence != null ? `${(p.confidence * 100).toFixed(1)}%` : "—"}
                    </span>
                    <span className="text-sm text-muted-foreground hidden md:block whitespace-nowrap">
                      {new Date(p.createdAt).toLocaleDateString()}
                    </span>
                  </div>
                ))}
              </div>
            </>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
