import { NextResponse } from "next/server"
import { prisma } from "@/lib/prisma"

/**
 * GET /api/metrics
 * Returns the most recent ModelMetrics record.
 * Used by the analytics dashboard to display Accuracy, FAR, FRR charts.
 */
export async function GET() {
  try {
    const metrics = await prisma.modelMetrics.findFirst({
      orderBy: { recordedAt: "desc" },
    })

    if (!metrics) {
      return NextResponse.json({ error: "No metrics found — run prisma db seed" }, { status: 404 })
    }

    return NextResponse.json({
      metrics: {
        accuracy: metrics.accuracy,
        far: metrics.far,
        frr: metrics.frr,
        f1Score: metrics.f1Score,
        precision: metrics.precision,
        recall: metrics.recall,
        totalSamples: metrics.totalSamples,
        tp: metrics.tp,
        tn: metrics.tn,
        fp: metrics.fp,
        fn: metrics.fn,
        dataSource: metrics.dataSource,
        recordedAt: metrics.recordedAt,
      },
    })
  } catch (error) {
    console.error("[GET /api/metrics]", error)
    return NextResponse.json({ error: "Failed to fetch metrics" }, { status: 500 })
  }
}
