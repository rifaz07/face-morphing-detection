import { NextResponse } from "next/server"
import { prisma } from "@/lib/prisma"

/**
 * GET /api/stats
 * Returns aggregated prediction statistics for the demo user.
 * Used by the dashboard to show quick summary cards.
 *
 * Response: { total, real, morphed, pending, errors, accuracy, avgConfidence }
 * TODO: Replace demo user with Clerk authenticated user.
 */
export async function GET() {
  try {
    const user = await prisma.user.findUnique({ where: { clerkId: "demo_user" } })
    if (!user) {
      return NextResponse.json(
        { error: "Demo user not found — run prisma db seed" },
        { status: 404 }
      )
    }

    const predictions = await prisma.prediction.findMany({
      where: { userId: user.id },
      select: { result: true, confidence: true },
    })

    const total = predictions.length
    const real = predictions.filter((p) => p.result === "REAL").length
    const morphed = predictions.filter((p) => p.result === "MORPHED").length
    const pending = predictions.filter((p) => p.result === "PENDING").length
    const errors = predictions.filter((p) => p.result === "ERROR").length

    const classified = predictions.filter(
      (p) => p.result === "REAL" || p.result === "MORPHED"
    )
    const accuracy = classified.length > 0 ? real / classified.length : 0

    const withConfidence = predictions.filter((p) => p.confidence !== null)
    const avgConfidence =
      withConfidence.length > 0
        ? withConfidence.reduce((sum, p) => sum + p.confidence, 0) / withConfidence.length
        : 0

    return NextResponse.json({
      total,
      real,
      morphed,
      pending,
      errors,
      accuracy: Math.round(accuracy * 10000) / 10000,
      avgConfidence: Math.round(avgConfidence * 10000) / 10000,
    })
  } catch (error) {
    console.error("[GET /api/stats]", error)
    return NextResponse.json({ error: "Failed to fetch stats" }, { status: 500 })
  }
}
