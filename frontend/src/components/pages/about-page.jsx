"use client";

import { motion } from "framer-motion";
import { Shield, Lightbulb, Code2, Users, BookOpen, ExternalLink } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import Link from "next/link";

const fadeUp = { hidden: { opacity: 0, y: 28 }, show: { opacity: 1, y: 0 } };

function Section({ children, className = "" }) {
  return (
    <motion.section
      initial="hidden"
      whileInView="show"
      viewport={{ once: true, margin: "-80px" }}
      variants={{ show: { transition: { staggerChildren: 0.12 } } }}
      className={`py-16 sm:py-20 ${className}`}
    >
      {children}
    </motion.section>
  );
}

const TECH_BADGES = [
  "Next.js 15", "FastAPI", "Python 3.11", "OpenCV", "scikit-learn",
  "scikit-image", "PostgreSQL", "Prisma ORM", "SQLAlchemy", "Docker",
  "Cloudinary", "Clerk", "Vercel", "Render", "Supabase", "Tailwind CSS v4",
];

const TEAM = [
  { name: "Student Name (Placeholder)", role: "Developer & Researcher", note: "B.Tech Computer Science · Final Year" },
  { name: "Mentor Name (Placeholder)", role: "Project Guide", note: "Department of Computer Science" },
];

export function AboutPage() {
  return (
    <main className="mx-auto max-w-4xl px-4 sm:px-6 lg:px-8 pb-20">

      {/* Hero */}
      <Section className="pt-16 sm:pt-24 text-center">
        <motion.div variants={fadeUp} transition={{ duration: 0.5 }}>
          <Badge variant="outline" className="mb-4">About the Project</Badge>
          <h1 className="text-4xl sm:text-5xl font-bold tracking-tight gradient-brand-text">
            Defending Biometric Integrity
          </h1>
          <p className="mt-5 text-muted-foreground text-lg leading-relaxed max-w-2xl mx-auto">
            A final-year research project combining classical computer vision
            and unsupervised machine learning to fight face morphing attacks in
            real-world biometric systems.
          </p>
        </motion.div>
      </Section>

      <Separator />

      {/* The Problem */}
      <Section>
        <motion.div variants={fadeUp} transition={{ duration: 0.5 }} className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 rounded-xl gradient-brand flex items-center justify-center shrink-0">
            <Shield className="size-5 text-white" />
          </div>
          <h2 className="text-2xl sm:text-3xl font-bold">The Problem</h2>
        </motion.div>
        <div className="space-y-4 text-muted-foreground leading-relaxed">
          <motion.p variants={fadeUp} transition={{ duration: 0.5 }}>
            <strong className="text-foreground">Face morphing</strong> is a class of biometric attack where two face images
            are digitally blended to create a single composite photo that can fool
            facial recognition systems into matching two different individuals.
            Attackers use this to obtain identity documents — passports, ID cards —
            that grant access under either identity.
          </motion.p>
          <motion.p variants={fadeUp} transition={{ duration: 0.5 }}>
            The threat is well-documented: a 2021 EU study found that standard
            automated border control systems accepted morphed images in up to
            <strong className="text-foreground"> 70% of test cases</strong>. With
            biometric passports now standard in over 180 countries, the attack
            surface is enormous and growing.
          </motion.p>
          <motion.p variants={fadeUp} transition={{ duration: 0.5 }}>
            Existing detection methods are either too computationally expensive for
            real-time use or require deep learning infrastructure that isn't
            practical for most deployments. There is a clear need for a lightweight,
            accurate, explainable solution.
          </motion.p>
        </div>
      </Section>

      <Separator />

      {/* The Solution */}
      <Section>
        <motion.div variants={fadeUp} transition={{ duration: 0.5 }} className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 rounded-xl gradient-brand flex items-center justify-center shrink-0">
            <Lightbulb className="size-5 text-white" />
          </div>
          <h2 className="text-2xl sm:text-3xl font-bold">The Solution</h2>
        </motion.div>
        <div className="space-y-4 text-muted-foreground leading-relaxed">
          <motion.p variants={fadeUp} transition={{ duration: 0.5 }}>
            Our approach combines two complementary feature extraction techniques —
            <strong className="text-foreground"> Local Binary Patterns (LBP)</strong> for
            texture-domain analysis and <strong className="text-foreground">Discrete Cosine
            Transform (DCT)</strong> for frequency-domain analysis — into a fused feature
            vector that is then classified using <strong className="text-foreground">K-Means
            clustering</strong>.
          </motion.p>
          <motion.p variants={fadeUp} transition={{ duration: 0.5 }}>
            LBP captures micro-texture inconsistencies introduced at morphing boundaries
            that are invisible to the human eye but statistically detectable. DCT exposes
            frequency artifacts from JPEG compression and alpha-blending operations
            common in software-based morphing tools.
          </motion.p>
          <motion.p variants={fadeUp} transition={{ duration: 0.5 }}>
            Using unsupervised K-Means rather than a supervised deep learning classifier
            keeps the model explainable, lightweight, and easy to audit — critical
            requirements for security-critical applications. The pipeline runs in under
            2 seconds on a standard server instance.
          </motion.p>
        </div>
      </Section>

      <Separator />

      {/* Methodology overview */}
      <Section>
        <motion.div variants={fadeUp} transition={{ duration: 0.5 }} className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 rounded-xl gradient-brand flex items-center justify-center shrink-0">
            <BookOpen className="size-5 text-white" />
          </div>
          <h2 className="text-2xl sm:text-3xl font-bold">Methodology</h2>
        </motion.div>
        <motion.p variants={fadeUp} transition={{ duration: 0.5 }} className="text-muted-foreground leading-relaxed mb-6">
          The detection pipeline has 8 discrete modules — from raw image input to
          final classification with confidence score and FAR/FRR evaluation.
        </motion.p>
        <motion.div variants={fadeUp} transition={{ duration: 0.5 }} className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {["Input Validation", "Face Detection", "Preprocessing", "LBP Extraction",
            "DCT Extraction", "Feature Fusion", "K-Means Clustering", "Classification"].map((m, i) => (
            <div key={m} className="rounded-xl border bg-card p-3 text-center">
              <span className="text-2xl font-bold gradient-brand-text">{String(i + 1).padStart(2, "0")}</span>
              <p className="text-xs text-muted-foreground mt-1 leading-tight">{m}</p>
            </div>
          ))}
        </motion.div>
        <motion.div variants={fadeUp} transition={{ duration: 0.5 }} className="mt-6">
          <Link href="/how-it-works" className="inline-flex items-center gap-1.5 text-sm font-medium text-muted-foreground hover:text-foreground transition-colors">
            Deep-dive into each module <ExternalLink className="size-3.5" />
          </Link>
        </motion.div>
      </Section>

      <Separator />

      {/* Team */}
      <Section>
        <motion.div variants={fadeUp} transition={{ duration: 0.5 }} className="flex items-center gap-3 mb-8">
          <div className="w-10 h-10 rounded-xl gradient-brand flex items-center justify-center shrink-0">
            <Users className="size-5 text-white" />
          </div>
          <h2 className="text-2xl sm:text-3xl font-bold">The Team</h2>
        </motion.div>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {TEAM.map(({ name, role, note }) => (
            <motion.div key={name} variants={fadeUp} transition={{ duration: 0.5 }}
              className="rounded-xl border bg-card p-6 flex flex-col gap-1"
            >
              <div className="w-10 h-10 rounded-full gradient-brand flex items-center justify-center text-white font-bold text-lg mb-2">
                {name[0]}
              </div>
              <p className="font-semibold">{name}</p>
              <p className="text-sm text-muted-foreground">{role}</p>
              <p className="text-xs text-muted-foreground">{note}</p>
            </motion.div>
          ))}
        </div>
        <motion.p variants={fadeUp} transition={{ duration: 0.5 }} className="mt-4 text-xs text-muted-foreground">
          College Name · Department of Computer Science · Academic Year 2024–25
        </motion.p>
      </Section>

      <Separator />

      {/* Tech Stack */}
      <Section>
        <motion.div variants={fadeUp} transition={{ duration: 0.5 }} className="flex items-center gap-3 mb-8">
          <div className="w-10 h-10 rounded-xl gradient-brand flex items-center justify-center shrink-0">
            <Code2 className="size-5 text-white" />
          </div>
          <h2 className="text-2xl sm:text-3xl font-bold">Tech Stack</h2>
        </motion.div>
        <motion.div variants={fadeUp} transition={{ duration: 0.5 }} className="flex flex-wrap gap-2">
          {TECH_BADGES.map((tech) => (
            <Badge key={tech} variant="secondary" className="text-xs px-3 py-1">
              {tech}
            </Badge>
          ))}
        </motion.div>
      </Section>
    </main>
  );
}
