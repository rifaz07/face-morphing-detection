import { LandingPage } from "@/components/pages/landing-page";

export const metadata = {
  title: "Face Morphing Detection — Detect Morphed Faces with AI",
  description:
    "Upload a face image and instantly detect morphing attacks using our LBP + DCT + K-Means ML pipeline. Production-grade biometric security for your applications.",
};

export default function HomePage() {
  return <LandingPage />;
}
