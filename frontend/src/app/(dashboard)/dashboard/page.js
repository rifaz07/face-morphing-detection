"use client";

import { useEffect, useState } from "react";
import Image from "next/image";
import Link from "next/link";
import { useUser } from "@clerk/nextjs";
import { Shield, ScanFace, Activity, TrendingUp, ArrowRight, Upload } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";

function StatCard({ title, value, subtitle, icon: Icon, loading }) {
  return (
    <Card className="border-border/50 bg-card/80 backdrop-blur-sm">
      <CardHeader className="flex flex-row items-center justify-between pb-2">
        <CardTitle className="text-sm font-medium text-muted-foreground">{title}</CardTitle>
        <Icon className="size-4 text-muted-foreground" aria-hidden />
      </CardHeader>
      <CardContent>
        {loading ? (
          <Skeleton className="h-8 w-20" />
        ) : (
          <p className="text-3xl font-bold">{value}</p>
        )}
        {subtitle && <p className="mt-1 text-xs text-muted-foreground">{subtitle}</p>}
      </CardContent>
    </Card>
  );
}

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

export default function DashboardPage() {
  const { user } = useUser();
  const [stats, setStats] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchData() {
      try {
        const [statsRes, predsRes] = await Promise.all([
          fetch("/api/stats"),
          fetch("/api/predictions"),
        ]);
        if (statsRes.ok) setStats(await statsRes.json());
        if (predsRes.ok) {
          const data = await predsRes.json();
          setPredictions(data.predictions?.slice(0, 5) ?? []);
        }
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, []);

  const firstName = user?.firstName ?? "there";

  return (
    <div className="p-6 md:p-8 max-w-6xl mx-auto space-y-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold">Welcome back, {firstName}!</h1>
          <p className="text-muted-foreground mt-1">
            Here&apos;s a summary of your detection activity.
          </p>
        </div>
        <Button
          className="gradient-brand text-white hover:opacity-90 w-fit gap-2"
          asChild
        >
          <Link href="/detect">
            <ScanFace className="size-4" />
            Start Detection
          </Link>
        </Button>
      </div>

      {/* Stat cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Total Detections"
          value={stats?.total ?? 0}
          subtitle="All-time uploads"
          icon={Activity}
          loading={loading}
        />
        <StatCard
          title="Real Faces"
          value={stats?.real ?? 0}
          subtitle="Authentic images"
          icon={Shield}
          loading={loading}
        />
        <StatCard
          title="Morphed Faces"
          value={stats?.morphed ?? 0}
          subtitle="Detected attacks"
          icon={ScanFace}
          loading={loading}
        />
        <StatCard
          title="Avg Confidence"
          value={
            stats
              ? `${(stats.avgConfidence * 100).toFixed(1)}%`
              : "—"
          }
          subtitle="Model certainty"
          icon={TrendingUp}
          loading={loading}
        />
      </div>

      {/* Quick detect card */}
      <Card className="border-dashed border-purple-500/30 bg-purple-500/5">
        <CardContent className="py-6 flex flex-col sm:flex-row items-center gap-4">
          <div className="rounded-full bg-purple-500/10 p-3 shrink-0">
            <Upload className="size-5 text-purple-500" aria-hidden />
          </div>
          <div className="flex-1 text-center sm:text-left">
            <p className="font-semibold">Quick Detect</p>
            <p className="text-sm text-muted-foreground">
              Upload a face image to instantly check if it&apos;s real or morphed.
            </p>
          </div>
          <Button
            size="sm"
            className="gradient-brand text-white hover:opacity-90 gap-2 shrink-0"
            asChild
          >
            <Link href="/detect">
              <ScanFace className="size-4" />
              Detect Now
            </Link>
          </Button>
        </CardContent>
      </Card>

      {/* Recent predictions */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold">Recent Detections</h2>
          <Button variant="ghost" size="sm" className="gap-1 text-sm" asChild>
            <Link href="/history">
              View all <ArrowRight className="size-3.5" />
            </Link>
          </Button>
        </div>

        {loading ? (
          <div className="space-y-3">
            {Array.from({ length: 3 }).map((_, i) => (
              <Skeleton key={i} className="h-14 w-full rounded-lg" />
            ))}
          </div>
        ) : predictions.length === 0 ? (
          <Card className="border-dashed border-border/50 bg-muted/30">
            <CardContent className="py-12 flex flex-col items-center gap-3 text-center">
              <ScanFace className="size-10 text-muted-foreground/50" aria-hidden />
              <p className="text-muted-foreground text-sm">No detections yet.</p>
              <Button
                size="sm"
                className="gradient-brand text-white hover:opacity-90"
                asChild
              >
                <Link href="/detect">Upload your first image</Link>
              </Button>
            </CardContent>
          </Card>
        ) : (
          <Card className="border-border/50 overflow-hidden">
            <div className="divide-y divide-border/50">
              {predictions.map((p) => (
                <div
                  key={p.id}
                  className="flex items-center justify-between px-5 py-3.5"
                >
                  <div className="flex items-center gap-3">
                    {/* Thumbnail */}
                    <div className="size-9 rounded-lg overflow-hidden bg-muted shrink-0">
                      {p.imageUrl ? (
                        <Image
                          src={p.imageUrl}
                          alt={p.imageName}
                          width={36}
                          height={36}
                          className="object-cover size-9"
                          unoptimized={false}
                        />
                      ) : (
                        <div className="size-9 flex items-center justify-center">
                          <ScanFace className="size-4 text-muted-foreground" />
                        </div>
                      )}
                    </div>
                    <div>
                      <p className="text-sm font-medium truncate max-w-[180px]">
                        {p.imageName}
                      </p>
                      <p className="text-xs text-muted-foreground">
                        {new Date(p.createdAt).toLocaleDateString()}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    {p.confidence != null && (
                      <span className="text-xs text-muted-foreground hidden sm:block">
                        {(p.confidence * 100).toFixed(1)}%
                      </span>
                    )}
                    <ResultBadge result={p.result} />
                  </div>
                </div>
              ))}
            </div>
          </Card>
        )}
      </div>
    </div>
  );
}
