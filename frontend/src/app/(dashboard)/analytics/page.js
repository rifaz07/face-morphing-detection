"use client";

import { useEffect, useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { BarChart2, Cpu, Layers } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";

const METRIC_COLORS = {
  Accuracy: "#7c3aed",
  FAR: "#ef4444",
  FRR: "#f97316",
};

function MetricCard({ label, value, description, loading }) {
  return (
    <Card className="border-border/50">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium text-muted-foreground">{label}</CardTitle>
      </CardHeader>
      <CardContent>
        {loading ? (
          <Skeleton className="h-10 w-24" />
        ) : (
          <p className="text-4xl font-bold">{value}</p>
        )}
        <p className="text-xs text-muted-foreground mt-1">{description}</p>
      </CardContent>
    </Card>
  );
}

export default function AnalyticsPage() {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/api/metrics")
      .then((r) => r.json())
      .then((d) => setMetrics(d.metrics ?? null))
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  const chartData = metrics
    ? [
        { name: "Accuracy", value: parseFloat((metrics.accuracy * 100).toFixed(1)) },
        { name: "FAR", value: parseFloat((metrics.far * 100).toFixed(1)) },
        { name: "FRR", value: parseFloat((metrics.frr * 100).toFixed(1)) },
      ]
    : [];

  return (
    <div className="p-6 md:p-8 max-w-5xl mx-auto space-y-8">
      <div>
        <h1 className="text-2xl font-bold">Model Analytics</h1>
        <p className="text-muted-foreground mt-1">
          Performance metrics for the K-Means clustering model.
        </p>
      </div>

      {/* Metric cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <MetricCard
          label="Accuracy"
          value={metrics ? `${(metrics.accuracy * 100).toFixed(1)}%` : "—"}
          description="Correct classifications"
          loading={loading}
        />
        <MetricCard
          label="FAR"
          value={metrics ? `${(metrics.far * 100).toFixed(1)}%` : "—"}
          description="False Acceptance Rate — morphed passed as real"
          loading={loading}
        />
        <MetricCard
          label="FRR"
          value={metrics ? `${(metrics.frr * 100).toFixed(1)}%` : "—"}
          description="False Rejection Rate — real rejected as morphed"
          loading={loading}
        />
      </div>

      {/* Bar chart */}
      <Card className="border-border/50">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <BarChart2 className="size-4 text-muted-foreground" aria-hidden />
            Metrics Overview
          </CardTitle>
        </CardHeader>
        <CardContent>
          {loading ? (
            <Skeleton className="h-56 w-full rounded-lg" />
          ) : (
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={chartData} barSize={48}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis
                  dataKey="name"
                  tick={{ fontSize: 13, fill: "hsl(var(--muted-foreground))" }}
                  axisLine={false}
                  tickLine={false}
                />
                <YAxis
                  domain={[0, 100]}
                  tickFormatter={(v) => `${v}%`}
                  tick={{ fontSize: 12, fill: "hsl(var(--muted-foreground))" }}
                  axisLine={false}
                  tickLine={false}
                />
                <Tooltip
                  formatter={(v) => [`${v}%`, ""]}
                  contentStyle={{
                    background: "hsl(var(--card))",
                    border: "1px solid hsl(var(--border))",
                    borderRadius: "0.5rem",
                    fontSize: 13,
                  }}
                />
                <Bar dataKey="value" radius={[6, 6, 0, 0]}>
                  {chartData.map((entry) => (
                    <Cell key={entry.name} fill={METRIC_COLORS[entry.name] ?? "#7c3aed"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
        </CardContent>
      </Card>

      {/* Model info */}
      <Card className="border-border/50 bg-muted/20">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <Cpu className="size-4 text-muted-foreground" aria-hidden />
            Model Information
          </CardTitle>
        </CardHeader>
        <CardContent>
          <dl className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-sm">
            {[
              { label: "Algorithm", value: "K-Means Clustering" },
              { label: "Feature Dim", value: "1,083" },
              { label: "LBP Features", value: "59" },
              { label: "DCT Features", value: "1,024" },
            ].map(({ label, value }) => (
              <div key={label}>
                <dt className="text-muted-foreground">{label}</dt>
                <dd className="font-medium mt-0.5">{value}</dd>
              </div>
            ))}
          </dl>
          <div className="mt-4 flex items-center gap-2 text-xs text-muted-foreground">
            <Layers className="size-3.5 shrink-0" aria-hidden />
            Trained on synthetic morphed/real face dataset. Production model pending real-world calibration.
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
