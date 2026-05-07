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

    return NextResponse.json({ metrics })
  } catch (error) {
    console.error("[GET /api/metrics]", error)
    return NextResponse.json({ error: "Failed to fetch metrics" }, { status: 500 })
  }
}
