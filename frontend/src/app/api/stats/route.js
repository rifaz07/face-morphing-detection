import { auth, currentUser } from '@clerk/nextjs/server'
import { prisma } from '@/lib/prisma'
import { NextResponse } from 'next/server'

/**
 * Find or auto-create the DB user for the authenticated Clerk userId.
 */
async function getOrCreateDbUser(userId) {
  let user = await prisma.user.findUnique({ where: { clerkId: userId } })
  if (!user) {
    const clerkUser = await currentUser()
    user = await prisma.user.upsert({
      where: { clerkId: userId },
      update: {},
      create: {
        clerkId: userId,
        email: clerkUser.emailAddresses[0].emailAddress,
        name: `${clerkUser.firstName ?? ''} ${clerkUser.lastName ?? ''}`.trim(),
        imageUrl: clerkUser.imageUrl,
        role: 'USER',
      },
    })
  }
  return user
}

/**
 * GET /api/stats
 * Returns aggregated prediction statistics for the authenticated user.
 * Response: { total, real, morphed, pending, errors, accuracy, avgConfidence }
 */
export async function GET() {
  try {
    const { userId } = await auth()
    if (!userId) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
    }

    const dbUser = await getOrCreateDbUser(userId)

    const predictions = await prisma.prediction.findMany({
      where: { userId: dbUser.id },
      select: { result: true, confidence: true },
    })

    const total = predictions.length
    const real = predictions.filter((p) => p.result === 'REAL').length
    const morphed = predictions.filter((p) => p.result === 'MORPHED').length
    const pending = predictions.filter((p) => p.result === 'PENDING').length
    const errors = predictions.filter((p) => p.result === 'ERROR').length

    const classified = predictions.filter(
      (p) => p.result === 'REAL' || p.result === 'MORPHED'
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
    console.error('[GET /api/stats]', error)
    return NextResponse.json({ error: 'Failed to fetch stats' }, { status: 500 })
  }
}
