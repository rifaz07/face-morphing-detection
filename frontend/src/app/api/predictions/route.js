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
 * GET /api/predictions
 * Returns predictions for the currently authenticated user, most recent first.
 * Limit: 50 records.
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
      orderBy: { createdAt: 'desc' },
      take: 50,
      include: {
        user: { select: { id: true, name: true, email: true } },
      },
    })

    return NextResponse.json({ predictions, count: predictions.length })
  } catch (error) {
    console.error('[GET /api/predictions]', error)
    return NextResponse.json({ error: 'Failed to fetch predictions' }, { status: 500 })
  }
}

/**
 * POST /api/predictions
 * Creates a new prediction record for the authenticated user.
 * Note: in normal flow, predictions are created by /api/detect.
 * This route remains as a fallback for direct creates.
 */
export async function POST(request) {
  try {
    const { userId } = await auth()
    if (!userId) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
    }

    const dbUser = await getOrCreateDbUser(userId)
    const body = await request.json()

    const {
      imageUrl,
      imageName,
      imageSize,
      result = 'PENDING',
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
        { error: 'imageUrl, imageName, and imageSize are required' },
        { status: 400 }
      )
    }

    const prediction = await prisma.prediction.create({
      data: {
        userId: dbUser.id,
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
    console.error('[POST /api/predictions]', error)
    return NextResponse.json({ error: 'Failed to create prediction' }, { status: 500 })
  }
}
