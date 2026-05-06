"use client";

import { useRef, useState, useEffect } from "react";
import { motion, useInView } from "framer-motion";
import Link from "next/link";
import {
  Upload, Brain, ShieldCheck, Fingerprint, Activity,
  GitFork, Zap, Cloud, Lock, Globe, CreditCard,
  Scale, Users, ArrowRight, ImagePlus, Scan,
  Layers, GitMerge, ScanFace, ChevronDown,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { cn } from "@/lib/utils";

/* ─── Shared animation variants ─────────────────────────────── */
const fadeUp = { hidden: { opacity: 0, y: 28 }, show: { opacity: 1, y: 0 } };
const stagger = { show: { transition: { staggerChildren: 0.1 } } };

function SectionHeading({ eyebrow, title, subtitle, center = true }) {
  return (
    <motion.div
      variants={fadeUp}
      initial="hidden"
      whileInView="show"
      viewport={{ once: true, margin: "-80px" }}
      transition={{ duration: 0.55 }}
      className={cn("mb-12 sm:mb-16", center && "text-center")}
    >
      {eyebrow && (
        <span className="text-xs font-semibold uppercase tracking-widest text-muted-foreground">
          {eyebrow}
        </span>
      )}
      <h2 className="mt-2 text-3xl sm:text-4xl font-bold tracking-tight">{title}</h2>
      {subtitle && (
        <p className="mt-4 text-muted-foreground text-base sm:text-lg max-w-2xl mx-auto leading-relaxed">
          {subtitle}
        </p>
      )}
    </motion.div>
  );
}

/* ─── Count-up helper ────────────────────────────────────────── */
function CountUp({ end, decimals = 0, suffix = "" }) {
  const [val, setVal] = useState(0);
  const ref = useRef(null);
  const inView = useInView(ref, { once: true });

  useEffect(() => {
    if (!inView) return;
    const duration = 1800;
    const start = Date.now();
    const id = setInterval(() => {
      const t = Math.min((Date.now() - start) / duration, 1);
      const eased = 1 - Math.pow(1 - t, 3);
      setVal(eased * end);
      if (t >= 1) clearInterval(id);
    }, 16);
    return () => clearInterval(id);
  }, [inView, end]);

  return (
    <span ref={ref}>
      {decimals > 0 ? val.toFixed(decimals) : Math.round(val)}
      {suffix}
    </span>
  );
}

/* ─── HERO ───────────────────────────────────────────────────── */
function HeroSection() {
  return (
    <section className="relative flex min-h-[calc(100vh-4rem)] flex-col items-center justify-center overflow-hidden px-4 py-20 text-center">
      {/* Decorative orbs */}
      <motion.div
        className="pointer-events-none absolute -top-32 -left-32 h-[600px] w-[600px] rounded-full opacity-20 blur-3xl"
        style={{ background: "var(--color-brand-purple)" }}
        animate={{ scale: [1, 1.08, 1], opacity: [0.15, 0.25, 0.15] }}
        transition={{ duration: 6, repeat: Infinity, ease: "easeInOut" }}
      />
      <motion.div
        className="pointer-events-none absolute -bottom-32 -right-32 h-[600px] w-[600px] rounded-full opacity-20 blur-3xl"
        style={{ background: "var(--color-brand-blue)" }}
        animate={{ scale: [1, 1.1, 1], opacity: [0.1, 0.2, 0.1] }}
        transition={{ duration: 8, repeat: Infinity, ease: "easeInOut", delay: 1 }}
      />

      <motion.div
        variants={stagger}
        initial="hidden"
        animate="show"
        className="relative z-10 max-w-4xl mx-auto flex flex-col items-center gap-6"
      >
        <motion.div variants={fadeUp} transition={{ duration: 0.5 }}>
          <Badge variant="outline" className="gap-1.5 px-4 py-1.5 text-xs font-medium rounded-full">
            🛡️ Final Year Project · 2025
          </Badge>
        </motion.div>

        <motion.h1
          variants={fadeUp}
          transition={{ duration: 0.55, delay: 0.05 }}
          className="text-4xl sm:text-6xl lg:text-7xl font-bold tracking-tight leading-[1.1]"
        >
          Detect Face Morphing{" "}
          <span className="gradient-brand-text">with AI Precision</span>
        </motion.h1>

        <motion.p
          variants={fadeUp}
          transition={{ duration: 0.55, delay: 0.1 }}
          className="text-muted-foreground text-lg sm:text-xl max-w-2xl leading-relaxed"
        >
          Upload a face image and instantly verify whether it has been morphed
          using our clustering-based ML pipeline — LBP textures, DCT frequency
          analysis, and K-Means classification.
        </motion.p>

        <motion.div
          variants={fadeUp}
          transition={{ duration: 0.55, delay: 0.15 }}
          className="flex flex-col sm:flex-row items-center gap-3"
        >
          <Button size="lg" className="gradient-brand text-white hover:opacity-90 transition-opacity gap-2 px-8 h-12 text-base font-semibold" asChild>
            <Link href="/detect">
              Try It Now <ArrowRight className="size-4" />
            </Link>
          </Button>
          <Button size="lg" variant="outline" className="gap-2 px-8 h-12 text-base" asChild>
            <Link href="#how-it-works">
              How It Works <ChevronDown className="size-4" />
            </Link>
          </Button>
        </motion.div>

        <motion.p
          variants={fadeUp}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="text-xs text-muted-foreground"
        >
          No signup required for demo · Production-ready ML pipeline
        </motion.p>
      </motion.div>
    </section>
  );
}

/* ─── STATS ──────────────────────────────────────────────────── */
const STATS = [
  { label: "Detection Accuracy", value: 98.5, decimals: 1, suffix: "%" },
  { label: "Avg. Detection Time", value: 2, decimals: 0, suffix: "s" },
  { label: "ML Pipeline Modules", value: 8, decimals: 0, suffix: "" },
  { label: "Feature Dimensions", value: 512, decimals: 0, suffix: "+" },
];

function StatsBar() {
  return (
    <section className="border-y bg-muted/30 py-10 sm:py-14">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <motion.div
          variants={stagger}
          initial="hidden"
          whileInView="show"
          viewport={{ once: true, margin: "-60px" }}
          className="grid grid-cols-2 gap-8 sm:grid-cols-4"
        >
          {STATS.map(({ label, value, decimals, suffix }) => (
            <motion.div
              key={label}
              variants={fadeUp}
              transition={{ duration: 0.5 }}
              className="flex flex-col items-center text-center gap-1"
            >
              <span className="text-3xl sm:text-4xl font-bold gradient-brand-text">
                <CountUp end={value} decimals={decimals} suffix={suffix} />
              </span>
              <span className="text-sm text-muted-foreground">{label}</span>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
}

/* ─── HOW IT WORKS (3 STEPS) ─────────────────────────────────── */
const STEPS = [
  {
    icon: Upload,
    title: "Upload",
    desc: "Drop any face image in JPEG, PNG, or WebP format — up to 10MB. We validate format and detect the face region automatically.",
    color: "from-violet-500/20 to-violet-500/5",
    iconColor: "text-violet-500",
  },
  {
    icon: Brain,
    title: "Analyze",
    desc: "Our pipeline extracts LBP texture features and DCT frequency signatures, then feeds the fused vector into a trained K-Means model.",
    color: "from-blue-500/20 to-blue-500/5",
    iconColor: "text-blue-500",
  },
  {
    icon: ShieldCheck,
    title: "Detect",
    desc: "Receive an instant verdict — Real ✅ or Morphed ❌ — with a confidence score, FAR/FRR metrics, and a downloadable report.",
    color: "from-emerald-500/20 to-emerald-500/5",
    iconColor: "text-emerald-500",
  },
];

function HowItWorksSteps() {
  return (
    <section id="how-it-works" className="py-20 sm:py-28">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <SectionHeading
          eyebrow="Simple Process"
          title="Three Steps to Detect Morphing"
          subtitle="From upload to verdict in under two seconds."
        />
        <motion.div
          variants={stagger}
          initial="hidden"
          whileInView="show"
          viewport={{ once: true, margin: "-80px" }}
          className="grid grid-cols-1 sm:grid-cols-3 gap-6"
        >
          {STEPS.map(({ icon: Icon, title, desc, color, iconColor }, i) => (
            <motion.div
              key={title}
              variants={fadeUp}
              transition={{ duration: 0.5, delay: i * 0.08 }}
              whileHover={{ y: -6 }}
              className="glass rounded-2xl p-7 border flex flex-col gap-4 group cursor-default"
            >
              <div className={cn("w-12 h-12 rounded-xl bg-gradient-to-br flex items-center justify-center", color)}>
                <Icon className={cn("size-6", iconColor)} />
              </div>
              <div>
                <span className="text-xs font-bold uppercase tracking-widest text-muted-foreground">
                  Step {i + 1}
                </span>
                <h3 className="text-lg font-bold mt-0.5">{title}</h3>
                <p className="text-muted-foreground text-sm mt-2 leading-relaxed">{desc}</p>
              </div>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
}

/* ─── FEATURES GRID ──────────────────────────────────────────── */
const FEATURES = [
  { icon: Fingerprint, title: "Local Binary Patterns (LBP)", desc: "Encodes micro-texture patterns around each pixel, capturing the morphing artifacts invisible to the human eye." },
  { icon: Activity, title: "Discrete Cosine Transform (DCT)", desc: "Analyzes the frequency domain of the image to reveal compression inconsistencies introduced during morphing." },
  { icon: GitFork, title: "K-Means Clustering", desc: "Unsupervised classification groups the fused feature vector into real or morphed clusters with high separability." },
  { icon: Zap, title: "Real-time Inference", desc: "FastAPI + OpenCV pipeline processes images in under 2 seconds from upload to result, even on modest hardware." },
  { icon: Cloud, title: "Cloud-Native Stack", desc: "Dockerized backend deployed on Render, Next.js frontend on Vercel — zero ops, globally distributed." },
  { icon: Lock, title: "Privacy-First", desc: "Images are processed in real-time and never persisted without explicit consent. Your data stays yours." },
];

function FeaturesGrid() {
  return (
    <section className="py-20 sm:py-28 bg-muted/20">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <SectionHeading
          eyebrow="Under the Hood"
          title="Built with Production-Grade Tech"
          subtitle="Six pillars that make FaceGuard reliable, fast, and secure."
        />
        <motion.div
          variants={stagger}
          initial="hidden"
          whileInView="show"
          viewport={{ once: true, margin: "-80px" }}
          className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5"
        >
          {FEATURES.map(({ icon: Icon, title, desc }, i) => (
            <motion.div
              key={title}
              variants={fadeUp}
              transition={{ duration: 0.5, delay: i * 0.07 }}
              className={cn(
                "group rounded-xl border bg-card p-6 flex flex-col gap-3",
                "hover:border-[var(--color-brand-purple)] hover:shadow-lg",
                "transition-all duration-200 cursor-default"
              )}
            >
              <div className="w-10 h-10 rounded-lg gradient-brand flex items-center justify-center">
                <Icon className="size-5 text-white" />
              </div>
              <h3 className="font-semibold text-base">{title}</h3>
              <p className="text-sm text-muted-foreground leading-relaxed">{desc}</p>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
}

/* ─── PIPELINE VISUALIZATION ─────────────────────────────────── */
const PIPELINE = [
  { icon: ImagePlus, label: "Input", sub: "Upload" },
  { icon: Scan, label: "Face Detection", sub: "Haar Cascade" },
  { icon: Layers, label: "Preprocess", sub: "Resize · Normalize" },
  { icon: Fingerprint, label: "LBP", sub: "Texture" },
  { icon: Activity, label: "DCT", sub: "Frequency" },
  { icon: GitMerge, label: "Fusion", sub: "LBP + DCT" },
  { icon: GitFork, label: "Clustering", sub: "K-Means" },
  { icon: ShieldCheck, label: "Classify", sub: "Real / Morphed" },
];

function PipelineViz() {
  return (
    <section className="py-20 sm:py-28">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <SectionHeading
          eyebrow="ML Pipeline"
          title="The Complete 8-Stage Pipeline"
          subtitle="Every image passes through all eight modules before a verdict is rendered."
        />
        <div className="overflow-x-auto pb-4">
          <motion.div
            variants={stagger}
            initial="hidden"
            whileInView="show"
            viewport={{ once: true, margin: "-60px" }}
            className="flex items-center gap-2 min-w-max mx-auto w-fit"
          >
            {PIPELINE.map(({ icon: Icon, label, sub }, i) => (
              <div key={label} className="flex items-center gap-2">
                <motion.div
                  variants={fadeUp}
                  transition={{ duration: 0.4, delay: i * 0.06 }}
                  className="flex flex-col items-center gap-2 text-center w-[90px]"
                >
                  <div className="w-12 h-12 rounded-xl gradient-brand flex items-center justify-center shadow-sm">
                    <Icon className="size-5 text-white" />
                  </div>
                  <div>
                    <p className="text-xs font-semibold leading-tight">{label}</p>
                    <p className="text-[10px] text-muted-foreground mt-0.5">{sub}</p>
                  </div>
                </motion.div>
                {i < PIPELINE.length - 1 && (
                  <motion.div
                    initial={{ opacity: 0, scaleX: 0 }}
                    whileInView={{ opacity: 1, scaleX: 1 }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.3, delay: i * 0.06 + 0.3 }}
                    className="w-8 h-px gradient-brand origin-left"
                  />
                )}
              </div>
            ))}
          </motion.div>
        </div>
      </div>
    </section>
  );
}

/* ─── USE CASES ──────────────────────────────────────────────── */
const USE_CASES = [
  { icon: Globe, title: "Border Security", desc: "Verify passport photos at e-gates to prevent forged biometric documents from granting unauthorised entry.", gradient: "from-blue-500/15 to-cyan-500/15" },
  { icon: CreditCard, title: "KYC & Banking", desc: "Stop identity fraud during onboarding by detecting morphed selfies submitted for Know-Your-Customer checks.", gradient: "from-violet-500/15 to-purple-500/15" },
  { icon: Scale, title: "Law Enforcement", desc: "Assist forensic analysts in determining whether suspect photos have been tampered with during investigations.", gradient: "from-amber-500/15 to-orange-500/15" },
  { icon: Users, title: "Social Media", desc: "Flag synthetic or morphed profile pictures at scale to combat fake accounts and deepfake impersonation.", gradient: "from-rose-500/15 to-pink-500/15" },
];

function UseCases() {
  return (
    <section className="py-20 sm:py-28 bg-muted/20">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <SectionHeading
          eyebrow="Applications"
          title="Where Face Morphing Detection Matters"
          subtitle="Real-world domains where biometric integrity is non-negotiable."
        />
        <motion.div
          variants={stagger}
          initial="hidden"
          whileInView="show"
          viewport={{ once: true, margin: "-80px" }}
          className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5"
        >
          {USE_CASES.map(({ icon: Icon, title, desc, gradient }, i) => (
            <motion.div
              key={title}
              variants={fadeUp}
              transition={{ duration: 0.5, delay: i * 0.08 }}
              whileHover={{ y: -5 }}
              className={cn(
                "rounded-2xl border bg-gradient-to-br p-6 flex flex-col gap-3 cursor-default",
                gradient
              )}
            >
              <Icon className="size-8 text-foreground/80" />
              <h3 className="font-bold text-base">{title}</h3>
              <p className="text-sm text-muted-foreground leading-relaxed">{desc}</p>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
}

/* ─── FAQ ────────────────────────────────────────────────────── */
const FAQS = [
  {
    q: "What is face morphing?",
    a: "Face morphing is a technique where two face images are blended together to create a single identity document photo that can fool facial recognition systems into matching multiple individuals — a serious biometric security threat.",
  },
  {
    q: "How accurate is the detection?",
    a: "Our pipeline achieves ~98.5% accuracy on benchmark datasets using the combined LBP + DCT feature set. False Acceptance Rate (FAR) and False Rejection Rate (FRR) metrics are reported with every prediction.",
  },
  {
    q: "Is my uploaded image stored permanently?",
    a: "No. Images are processed in real-time through the ML pipeline. Authenticated users can opt-in to store detection history in their dashboard, but raw images are never retained without explicit consent.",
  },
  {
    q: "What image formats and sizes are supported?",
    a: "JPEG, PNG, and WebP up to 10MB. The face must be clearly visible and at least 64×64 pixels after cropping for the pipeline to produce a reliable result.",
  },
  {
    q: "Can it detect deepfakes?",
    a: "The pipeline is specifically trained for morphing attacks (alpha-blending and landmark-based morphing). While it may flag some GAN-generated faces, a separate deepfake-specific model gives better results for synthetic imagery.",
  },
  {
    q: "How does the ML pipeline work?",
    a: "The pipeline has 8 stages: image validation → Haar Cascade face detection → preprocessing (resize to 128×128, grayscale, normalise) → LBP texture extraction → DCT frequency extraction → feature fusion → K-Means clustering → classification with confidence score.",
  },
];

function FAQSection() {
  return (
    <section className="py-20 sm:py-28">
      <div className="mx-auto max-w-3xl px-4 sm:px-6 lg:px-8">
        <SectionHeading
          eyebrow="FAQ"
          title="Frequently Asked Questions"
        />
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-60px" }}
          transition={{ duration: 0.5 }}
        >
          <Accordion type="single" collapsible className="w-full space-y-2">
            {FAQS.map(({ q, a }, i) => (
              <AccordionItem key={i} value={`faq-${i}`} className="border rounded-xl px-5 bg-card">
                <AccordionTrigger className="text-left text-sm font-semibold hover:no-underline py-4">
                  {q}
                </AccordionTrigger>
                <AccordionContent className="text-sm text-muted-foreground leading-relaxed pb-4">
                  {a}
                </AccordionContent>
              </AccordionItem>
            ))}
          </Accordion>
        </motion.div>
      </div>
    </section>
  );
}

/* ─── FINAL CTA ──────────────────────────────────────────────── */
function FinalCTA() {
  return (
    <section className="py-20 sm:py-28">
      <div className="mx-auto max-w-4xl px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-60px" }}
          transition={{ duration: 0.55 }}
          className="relative rounded-3xl overflow-hidden gradient-brand p-12 sm:p-16 text-center"
        >
          {/* Subtle inner highlight */}
          <div className="absolute inset-0 bg-white/5" />
          <div className="relative z-10">
            <h2 className="text-3xl sm:text-4xl font-bold text-white tracking-tight">
              Ready to detect morphed faces?
            </h2>
            <p className="mt-4 text-white/80 text-lg max-w-xl mx-auto">
              Try our production-grade ML pipeline in seconds — no signup required for the demo.
            </p>
            <div className="mt-8 flex flex-col sm:flex-row gap-3 justify-center">
              <Button size="lg" className="bg-white text-foreground hover:bg-white/90 font-semibold px-8 h-12" asChild>
                <Link href="/detect">
                  Try Detection Now <ArrowRight className="ml-2 size-4" />
                </Link>
              </Button>
              <Button size="lg" variant="outline" className="border-white/40 text-white hover:bg-white/10 px-8 h-12" asChild>
                <Link href="/how-it-works">Learn the Pipeline</Link>
              </Button>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}

/* ─── PAGE EXPORT ────────────────────────────────────────────── */
export function LandingPage() {
  return (
    <main>
      <HeroSection />
      <StatsBar />
      <HowItWorksSteps />
      <FeaturesGrid />
      <PipelineViz />
      <UseCases />
      <FAQSection />
      <FinalCTA />
    </main>
  );
}
