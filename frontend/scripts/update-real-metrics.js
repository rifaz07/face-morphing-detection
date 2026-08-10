/**
 * One-off script: update the ModelMetrics row with the real evaluation
 * numbers produced by ml/checkpoints/real_evaluation_report.json (the
 * 4,000-image Kaggle SSMD sample run). Does not re-run evaluation.
 *
 * Usage: node scripts/update-real-metrics.js
 */
const fs = require("fs")
const path = require("path")
const { PrismaClient } = require("@prisma/client")

const prisma = new PrismaClient()

const REPORT_PATH = path.join(
  __dirname,
  "..",
  "..",
  "ml",
  "checkpoints",
  "real_evaluation_report.json"
)

async function main() {
  const report = JSON.parse(fs.readFileSync(REPORT_PATH, "utf-8"))
  const m = report.metrics

  const existing = await prisma.modelMetrics.findFirst({
    orderBy: { recordedAt: "desc" },
  })

  const data = {
    accuracy: m.accuracy,
    far: m.far,
    frr: m.frr,
    f1Score: m.f1_score,
    precision: m.precision,
    recall: m.recall,
    totalSamples: m.total_samples,
    tp: m.tp,
    tn: m.tn,
    fp: m.fp,
    fn: m.fn,
    dataSource: "real",
  }

  const result = existing
    ? await prisma.modelMetrics.update({ where: { id: existing.id }, data })
    : await prisma.modelMetrics.create({ data })

  console.log(`✔ ModelMetrics row ${result.id} updated with real evaluation data:`)
  console.log(data)
}

main()
  .catch((err) => {
    console.error("❌ update-real-metrics failed:", err)
    process.exit(1)
  })
  .finally(() => prisma.$disconnect())
