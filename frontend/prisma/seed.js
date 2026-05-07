const { PrismaClient } = require("@prisma/client")

const prisma = new PrismaClient()

async function main() {
  console.log("🌱  Seeding database...")

  // ── Users ────────────────────────────────────────────────────────────────

  const admin = await prisma.user.upsert({
    where: { clerkId: "demo_admin" },
    update: {},
    create: {
      clerkId: "demo_admin",
      email: "admin@faceguard.dev",
      name: "Admin User",
      role: "ADMIN",
    },
  })

  const regularUser = await prisma.user.upsert({
    where: { clerkId: "demo_user" },
    update: {},
    create: {
      clerkId: "demo_user",
      email: "user@faceguard.dev",
      name: "Demo User",
      role: "USER",
    },
  })

  console.log(`  ✔ Users: admin (${admin.id}), user (${regularUser.id})`)

  // ── Predictions ──────────────────────────────────────────────────────────

  const predictions = [
    {
      userId: regularUser.id,
      imageUrl: "https://res.cloudinary.com/demo/image/upload/sample_face_1.jpg",
      imageName: "passport_photo_1.jpg",
      imageSize: 204800,
      result: "REAL",
      confidence: 0.923,
      faceCount: 1,
      processingTimeMs: 312.4,
      metadata: { width: 640, height: 480, format: "JPEG" },
    },
    {
      userId: regularUser.id,
      imageUrl: "https://res.cloudinary.com/demo/image/upload/sample_face_2.jpg",
      imageName: "id_card_photo.jpg",
      imageSize: 153600,
      result: "REAL",
      confidence: 0.871,
      faceCount: 1,
      processingTimeMs: 287.1,
      metadata: { width: 480, height: 640, format: "JPEG" },
    },
    {
      userId: regularUser.id,
      imageUrl: "https://res.cloudinary.com/demo/image/upload/sample_face_3.jpg",
      imageName: "selfie.png",
      imageSize: 512000,
      result: "REAL",
      confidence: 0.956,
      faceCount: 1,
      processingTimeMs: 341.8,
      metadata: { width: 800, height: 800, format: "PNG" },
    },
    {
      userId: regularUser.id,
      imageUrl: "https://res.cloudinary.com/demo/image/upload/morphed_face_1.jpg",
      imageName: "suspicious_doc.jpg",
      imageSize: 319488,
      result: "MORPHED",
      confidence: 0.812,
      faceCount: 1,
      processingTimeMs: 328.5,
      metadata: { width: 640, height: 480, format: "JPEG" },
    },
    {
      userId: regularUser.id,
      imageUrl: "https://res.cloudinary.com/demo/image/upload/morphed_face_2.jpg",
      imageName: "application_photo.jpg",
      imageSize: 245760,
      result: "MORPHED",
      confidence: 0.889,
      faceCount: 1,
      processingTimeMs: 295.3,
      metadata: { width: 500, height: 600, format: "JPEG" },
    },
  ]

  // Delete existing seed predictions to allow re-seeding
  await prisma.prediction.deleteMany({ where: { userId: regularUser.id } })

  for (const pred of predictions) {
    await prisma.prediction.create({ data: pred })
  }

  console.log(`  ✔ Predictions: 3 REAL, 2 MORPHED created for demo user`)

  // ── ModelMetrics ─────────────────────────────────────────────────────────

  await prisma.modelMetrics.create({
    data: {
      accuracy: 1.0,
      far: 0.0,
      frr: 0.0,
      f1Score: 1.0,
      precision: 1.0,
      recall: 1.0,
      totalSamples: 200,
      dataSource: "synthetic",
    },
  })

  console.log("  ✔ ModelMetrics: 1 evaluation record created")
  console.log("✅  Seeding complete!")
}

main()
  .catch((err) => {
    console.error("❌  Seed failed:", err)
    process.exit(1)
  })
  .finally(() => prisma.$disconnect())
