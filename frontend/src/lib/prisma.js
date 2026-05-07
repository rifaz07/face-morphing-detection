import { PrismaClient } from "@prisma/client"

/**
 * Singleton Prisma Client instance.
 *
 * Next.js hot-reload creates new module instances on every change, which
 * would exhaust the database connection pool. Attaching the client to
 * `globalThis` ensures we reuse the same instance across hot-reloads in
 * development while still creating a fresh client in production.
 */
const globalForPrisma = globalThis

export const prisma = globalForPrisma.prisma ?? new PrismaClient()

if (process.env.NODE_ENV !== "production") {
  globalForPrisma.prisma = prisma
}

export default prisma
