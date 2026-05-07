import { NextResponse } from "next/server"
import { prisma } from "@/lib/prisma"

/**
 * GET /api/predictions
 * Returns all predictions for the demo user.
 * TODO: Replace demo user lookup with Clerk auth (currentUser / auth())
 */
export async function GET() {
  try {
    const predictions = await prisma.prediction.findMany({
      where: { user: { clerkId: "demo_user" } },
      orderBy: { createdAt: "desc" },
      include: {
        user: { select: { id: true, name: true, email: true } },
      },
    })

    return NextResponse.json({ predictions, count: predictions.length })
  } catch (error) {
    console.error("[GET /api/predictions]", error)
    return NextResponse.json({ error: "Failed to fetch predictions" }, { status: 500 })
  }
}

/**
 * POST /api/predictions
 * Creates a new prediction record.
 *
 * Body: { imageUrl, imageName, imageSize, result, confidence, faceCount,
 *         processingTimeMs, lbpVector, dctVector, fusedVector, metadata }
 */
export async function POST(request) {
  try {
    const body = await request.json()

    const {
      imageUrl,
      imageName,
      imageSize,
      result = "PENDING",
      confidence,
      faceCount,
      processingTimeMs,
      lbpVector,
      dctVector,
      fusedVector,
      metadata,
    } = body

    if (!imageUrl || !imageName || !imageSize) {
      return NextResponse.json(
        { error: "imageUrl, imageName, and imageSize are required" },
        { status: 400 }
      )
    }

    // TODO: replace with authenticated user from Clerk
    const user = await prisma.user.findUnique({ where: { clerkId: "demo_user" } })
    if (!user) {
      return NextResponse.json({ error: "Demo user not found — run prisma db seed" }, { status: 404 })
    }

    const prediction = await prisma.prediction.create({
      data: {
        userId: user.id,
        imageUrl,
        imageName,
        imageSize,
        result,
        confidence,
        faceCount,
        processingTimeMs,
        lbpVector,
        dctVector,
        fusedVector,
        metadata,
      },
    })

    return NextResponse.json({ prediction }, { status: 201 })
  } catch (error) {
    console.error("[POST /api/predictions]", error)
    return NextResponse.json({ error: "Failed to create prediction" }, { status: 500 })
  }
}
