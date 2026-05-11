import { auth, currentUser } from '@clerk/nextjs/server'
import { prisma } from '@/lib/prisma'
import { NextResponse } from 'next/server'

/**
 * POST /api/auth/sync-user
 * Upserts the authenticated Clerk user into our Prisma database.
 * Called automatically on every dashboard page load via useSyncUser hook.
 */
export async function POST() {
  try {
    const { userId } = await auth()
    if (!userId) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
    }

    const clerkUser = await currentUser()

    const user = await prisma.user.upsert({
      where: { clerkId: userId },
      update: {
        email: clerkUser.emailAddresses[0].emailAddress,
        name: `${clerkUser.firstName ?? ''} ${clerkUser.lastName ?? ''}`.trim(),
        imageUrl: clerkUser.imageUrl,
      },
      create: {
        clerkId: userId,
        email: clerkUser.emailAddresses[0].emailAddress,
        name: `${clerkUser.firstName ?? ''} ${clerkUser.lastName ?? ''}`.trim(),
        imageUrl: clerkUser.imageUrl,
        role: 'USER',
      },
    })

    return NextResponse.json({ user })
  } catch (error) {
    console.error('[POST /api/auth/sync-user]', error)
    return NextResponse.json({ error: error.message }, { status: 500 })
  }
}
