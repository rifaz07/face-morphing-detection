import { auth, currentUser } from '@clerk/nextjs/server'
import { v2 as cloudinary } from 'cloudinary'
import { prisma } from '@/lib/prisma'
import { NextResponse } from 'next/server'

cloudinary.config({
  cloud_name: process.env.NEXT_PUBLIC_CLOUDINARY_CLOUD_NAME,
  api_key: process.env.CLOUDINARY_API_KEY,
  api_secret: process.env.CLOUDINARY_API_SECRET,
})

/**
 * Upload a buffer to Cloudinary and return the result.
 * @param {Buffer} buffer
 * @param {string} filename
 */
async function uploadToCloudinary(buffer, filename) {
  return new Promise((resolve, reject) => {
    cloudinary.uploader
      .upload_stream(
        {
          folder: 'faceguard/detections',
          resource_type: 'image',
          public_id: `${Date.now()}_${filename.replace(/\.[^.]+$/, '')}`,
        },
        (error, result) => {
          if (error) reject(error)
          else resolve(result)
        }
      )
      .end(buffer)
  })
}

/**
 * Find or create a DB user record for the authenticated Clerk user.
 */
async function getOrCreateDbUser(userId) {
  let dbUser = await prisma.user.findUnique({ where: { clerkId: userId } })
  if (!dbUser) {
    const clerkUser = await currentUser()
    dbUser = await prisma.user.upsert({
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
  return dbUser
}

/**
 * POST /api/detect
 * Full detection pipeline:
 *   1. Authenticate with Clerk
 *   2. Upload image to Cloudinary
 *   3. Send image to FastAPI ML pipeline
 *   4. Save result to Prisma
 *   5. Return detection result
 *
 * Accepts: FormData with `file` (image)
 * Returns: { prediction, confidence, faceCount, processingTimeMs, imageUrl, imageDimensions, predictionId }
 */
export async function POST(request) {
  try {
    // Step 1 — Auth
    const { userId } = await auth()
    if (!userId) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
    }

    // Step 2 — Parse file
    const formData = await request.formData()
    const file = formData.get('file')
    if (!file) {
      return NextResponse.json({ error: 'NO_FILE', message: 'No file provided' }, { status: 400 })
    }

    const arrayBuffer = await file.arrayBuffer()
    const buffer = Buffer.from(arrayBuffer)

    // Step 3 — Find/create DB user
    const dbUser = await getOrCreateDbUser(userId)

    // Step 4 — Upload to Cloudinary
    let cloudinaryResult
    try {
      cloudinaryResult = await uploadToCloudinary(buffer, file.name)
    } catch (err) {
      console.error('[detect] Cloudinary upload failed:', err)
      return NextResponse.json(
        { error: 'UPLOAD_FAILED', message: 'Failed to upload image to cloud storage' },
        { status: 500 }
      )
    }

    // Step 5 — Send to FastAPI ML pipeline
    const fastapiForm = new FormData()
    fastapiForm.append('file', new Blob([buffer], { type: file.type }), file.name)

    const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
    let fastapiRes
    try {
      fastapiRes = await fetch(`${apiUrl}/api/v1/detection/classify`, {
        method: 'POST',
        body: fastapiForm,
      })
    } catch (err) {
      console.error('[detect] FastAPI unreachable:', err)
      return NextResponse.json(
        { error: 'SERVICE_UNAVAILABLE', message: 'Detection service is currently unavailable' },
        { status: 503 }
      )
    }

    if (!fastapiRes.ok) {
      const errorBody = await fastapiRes.json().catch(() => ({}))
      const detail = errorBody?.detail ?? {}
      const errorCode = detail?.error_code ?? 'DETECTION_ERROR'
      const message = detail?.message ?? 'Detection failed'

      if (fastapiRes.status === 400) {
        return NextResponse.json({ error: 'INVALID_IMAGE', message }, { status: 400 })
      }
      return NextResponse.json({ error: errorCode, message }, { status: 500 })
    }

    const mlResult = await fastapiRes.json()

    // Step 6 — Handle no-face case
    if (!mlResult.prediction) {
      const saved = await prisma.prediction.create({
        data: {
          userId: dbUser.id,
          imageUrl: cloudinaryResult.secure_url,
          imageName: file.name,
          imageSize: file.size,
          result: 'ERROR',
          faceCount: mlResult.detection?.face_count ?? 0,
          errorMessage: 'No face detected in the image',
          metadata: {
            width: mlResult.detection?.image_dimensions?.width,
            height: mlResult.detection?.image_dimensions?.height,
            cloudinaryPublicId: cloudinaryResult.public_id,
          },
        },
      })
      return NextResponse.json(
        {
          error: 'NO_FACE',
          message: 'No face detected in the image. Please upload a clear face photo.',
          predictionId: saved.id,
          imageUrl: cloudinaryResult.secure_url,
        },
        { status: 422 }
      )
    }

    // Step 7 — Save to Prisma
    const totalMs =
      (mlResult.detection?.processing_time_ms ?? 0) +
      (mlResult.fusion?.processing_time_ms ?? 0) +
      (mlResult.prediction?.processing_time_ms ?? 0)

    const saved = await prisma.prediction.create({
      data: {
        userId: dbUser.id,
        imageUrl: cloudinaryResult.secure_url,
        imageName: file.name,
        imageSize: file.size,
        result: mlResult.prediction.prediction,
        confidence: mlResult.prediction.confidence,
        faceCount: mlResult.detection.face_count,
        processingTimeMs: Math.round(totalMs * 10) / 10,
        fusedVector: mlResult.fusion?.fused_vector ?? null,
        metadata: {
          width: mlResult.detection?.image_dimensions?.width,
          height: mlResult.detection?.image_dimensions?.height,
          cloudinaryPublicId: cloudinaryResult.public_id,
          cloudinaryFormat: cloudinaryResult.format,
          clusterId: mlResult.prediction.cluster_id,
          distanceToCentroid: mlResult.prediction.distance_to_centroid,
        },
      },
    })

    return NextResponse.json({
      prediction: mlResult.prediction.prediction,
      confidence: mlResult.prediction.confidence,
      faceCount: mlResult.detection.face_count,
      processingTimeMs: saved.processingTimeMs,
      imageUrl: cloudinaryResult.secure_url,
      imageDimensions: mlResult.detection.image_dimensions,
      predictionId: saved.id,
    })
  } catch (error) {
    console.error('[POST /api/detect]', error)
    return NextResponse.json({ error: 'INTERNAL_ERROR', message: error.message }, { status: 500 })
  }
}
