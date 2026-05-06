import { HowItWorksPage } from "@/components/pages/how-it-works-page";

export const metadata = {
  title: "How It Works",
  description:
    "A deep-dive into all 8 modules of the face morphing detection ML pipeline — from image upload to LBP, DCT, K-Means clustering, and final classification.",
};

export default function HowItWorks() {
  return <HowItWorksPage />;
}
