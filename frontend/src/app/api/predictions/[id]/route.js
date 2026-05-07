import { NextResponse } from "next/server"
import { prisma } from "@/lib/prisma"

/**
 * GET /api/predictions/:id
 * Returns a single prediction by its CUID.
 */
export async function GET(request, { params }) {
  try {
    const { id } = await params

    const prediction = await prisma.prediction.findUnique({
      where: { id },
      include: {
        user: { select: { id: true, name: true, email: true } },
      },
    })

    if (!prediction) {
      return NextResponse.json({ error: "Prediction not found" }, { status: 404 })
    }

    return NextResponse.json({ prediction })
  } catch (error) {
    console.error("[GET /api/predictions/:id]", error)
    return NextResponse.json({ error: "Failed to fetch prediction" }, { status: 500 })
  }
}

/**
 * DELETE /api/predictions/:id
 * Deletes a prediction by its CUID.
 */
export async function DELETE(request, { params }) {
  try {
    const { id } = await params

    const existing = await prisma.prediction.findUnique({ where: { id } })
    if (!existing) {
      return NextResponse.json({ error: "Prediction not found" }, { status: 404 })
    }

    await prisma.prediction.delete({ where: { id } })

    return NextResponse.json({ message: "Prediction deleted", id })
  } catch (error) {
    console.error("[DELETE /api/predictions/:id]", error)
    return NextResponse.json({ error: "Failed to delete prediction" }, { status: 500 })
  }
}
