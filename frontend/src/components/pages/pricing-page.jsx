"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Check, Zap, Star, Building2, ArrowRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { cn } from "@/lib/utils";
import Link from "next/link";

const fadeUp = { hidden: { opacity: 0, y: 28 }, show: { opacity: 1, y: 0 } };

const TIERS = [
  {
    icon: Zap,
    name: "Free",
    tagline: "Perfect for demos and evaluation",
    monthly: 0,
    cta: "Get Started",
    href: "/sign-up",
    popular: false,
    features: [
      "50 detections / month",
      "Basic accuracy report",
      "JPEG / PNG / WebP support",
      "Community support",
      "API playground access",
    ],
  },
  {
    icon: Star,
    name: "Pro",
    tagline: "For security teams and developers",
    monthly: 9,
    cta: "Start Free Trial",
    href: "/sign-up?plan=pro",
    popular: true,
    features: [
      "Unlimited detections",
      "Advanced analytics dashboard",
      "Confidence score + FAR/FRR",
      "PDF report export",
      "Priority email support",
      "Batch upload API",
    ],
  },
  {
    icon: Building2,
    name: "Enterprise",
    tagline: "Dedicated infrastructure for organisations",
    monthly: null,
    cta: "Contact Sales",
    href: "/contact",
    popular: false,
    features: [
      "Everything in Pro",
      "Dedicated API endpoint",
      "Custom SLA guarantee",
      "On-premise deployment option",
      "SSO / SAML support",
      "24 / 7 priority support",
    ],
  },
];

const PRICING_FAQS = [
  { q: "Is the free tier really free forever?", a: "Yes — 50 detections per month, no credit card required. The free tier is designed for demos, student projects, and evaluation." },
  { q: "Can I cancel the Pro plan anytime?", a: "Absolutely. Cancel at any time from your dashboard settings. You keep Pro access until the end of the current billing period." },
  { q: "What counts as one detection?", a: "Each image you submit to the /detect endpoint counts as one detection, regardless of the verdict returned." },
  { q: "Is there a student discount?", a: "Yes — verify your academic email at sign-up to receive 50% off the Pro plan. Contact us if your institution isn't automatically recognised." },
  { q: "Can I get a demo of the Enterprise plan?", a: "Yes. Use the Contact page to schedule a 30-minute call with our team and we'll set up a custom sandbox environment." },
];

export function PricingPage() {
  const [yearly, setYearly] = useState(false);

  return (
    <main className="pb-24">
      {/* Hero */}
      <section className="py-16 sm:py-24 text-center px-4">
        <motion.div
          initial={{ opacity: 0, y: 28 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.55 }}
          className="mx-auto max-w-2xl"
        >
          <Badge variant="outline" className="mb-4">Pricing</Badge>
          <h1 className="text-4xl sm:text-5xl font-bold tracking-tight">
            Simple, transparent pricing
          </h1>
          <p className="mt-4 text-muted-foreground text-lg">
            Start free. Scale when you need to. Cancel anytime.
          </p>

          {/* Monthly / Yearly toggle */}
          <div className="mt-8 inline-flex items-center gap-3 rounded-full border bg-muted/50 p-1.5 text-sm">
            <button
              onClick={() => setYearly(false)}
              className={cn(
                "px-4 py-1.5 rounded-full font-medium transition-all",
                !yearly ? "bg-background shadow text-foreground" : "text-muted-foreground"
              )}
            >
              Monthly
            </button>
            <button
              onClick={() => setYearly(true)}
              className={cn(
                "px-4 py-1.5 rounded-full font-medium transition-all flex items-center gap-1.5",
                yearly ? "bg-background shadow text-foreground" : "text-muted-foreground"
              )}
            >
              Yearly
              <Badge className="text-[10px] px-1.5 py-0 h-4 gradient-brand text-white border-0">
                −20%
              </Badge>
            </button>
          </div>
        </motion.div>
      </section>

      {/* Pricing cards */}
      <section className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <motion.div
          initial="hidden"
          whileInView="show"
          viewport={{ once: true, margin: "-60px" }}
          variants={{ show: { transition: { staggerChildren: 0.1 } } }}
          className="grid grid-cols-1 md:grid-cols-3 gap-6 items-stretch"
        >
          {TIERS.map(({ icon: Icon, name, tagline, monthly, cta, href, popular, features }) => {
            const price = monthly === null ? null : yearly ? Math.floor(monthly * 0.8) : monthly;
            return (
              <motion.div
                key={name}
                variants={fadeUp}
                transition={{ duration: 0.5 }}
                className={cn(
                  "relative rounded-2xl border p-8 flex flex-col gap-6",
                  popular
                    ? "border-[var(--color-brand-purple)] shadow-xl shadow-[var(--color-brand-purple)]/10"
                    : "bg-card"
                )}
              >
                {popular && (
                  <div className="absolute -top-3 left-1/2 -translate-x-1/2">
                    <Badge className="gradient-brand text-white border-0 shadow-sm px-4">
                      Most Popular
                    </Badge>
                  </div>
                )}

                <div>
                  <div className={cn("w-10 h-10 rounded-xl flex items-center justify-center mb-4", popular ? "gradient-brand" : "bg-muted")}>
                    <Icon className={cn("size-5", popular ? "text-white" : "text-muted-foreground")} />
                  </div>
                  <h2 className="text-xl font-bold">{name}</h2>
                  <p className="text-sm text-muted-foreground mt-1">{tagline}</p>
                </div>

                <div>
                  {price === null ? (
                    <p className="text-3xl font-bold">Custom</p>
                  ) : (
                    <div className="flex items-end gap-1">
                      <span className="text-4xl font-bold">${price}</span>
                      <span className="text-muted-foreground text-sm pb-1.5">/ {yearly ? "mo, billed yearly" : "month"}</span>
                    </div>
                  )}
                </div>

                <Button
                  className={cn(
                    "w-full gap-2",
                    popular ? "gradient-brand text-white hover:opacity-90" : ""
                  )}
                  variant={popular ? "default" : "outline"}
                  asChild
                >
                  <Link href={href}>
                    {cta} <ArrowRight className="size-4" />
                  </Link>
                </Button>

                <ul className="space-y-3 flex-1">
                  {features.map((f) => (
                    <li key={f} className="flex items-start gap-2.5 text-sm">
                      <Check className="size-4 mt-0.5 shrink-0 text-emerald-500" />
                      <span className="text-muted-foreground">{f}</span>
                    </li>
                  ))}
                </ul>
              </motion.div>
            );
          })}
        </motion.div>

        <motion.p
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5, delay: 0.3 }}
          className="text-center text-sm text-muted-foreground mt-8"
        >
          Cancel anytime · No credit card required for free tier · 14-day money-back guarantee on Pro
        </motion.p>
      </section>

      {/* FAQ */}
      <section className="mx-auto max-w-3xl px-4 sm:px-6 lg:px-8 mt-24">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-60px" }}
          transition={{ duration: 0.5 }}
          className="text-center mb-10"
        >
          <h2 className="text-2xl sm:text-3xl font-bold">Pricing FAQ</h2>
        </motion.div>
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
        >
          <Accordion type="single" collapsible className="space-y-2">
            {PRICING_FAQS.map(({ q, a }, i) => (
              <AccordionItem key={i} value={`pq-${i}`} className="border rounded-xl px-5 bg-card">
                <AccordionTrigger className="text-left text-sm font-semibold hover:no-underline py-4">{q}</AccordionTrigger>
                <AccordionContent className="text-sm text-muted-foreground leading-relaxed pb-4">{a}</AccordionContent>
              </AccordionItem>
            ))}
          </Accordion>
        </motion.div>
      </section>
    </main>
  );
}
