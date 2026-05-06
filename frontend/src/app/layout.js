import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { Providers } from "@/components/shared/providers";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata = {
  title: {
    default: "Face Morphing Detection System",
    template: "%s | Face Morphing Detection",
  },
  description:
    "Detect morphed face images in real-time using an AI-powered LBP + DCT + K-Means clustering pipeline. Protect biometric systems from face morphing attacks.",
  keywords: [
    "face morphing detection",
    "biometric security",
    "AI face analysis",
    "morphing attack detection",
    "LBP features",
    "DCT features",
    "K-Means clustering",
  ],
  authors: [{ name: "rifaz shaikh razak" }],
  creator: "rifaz shaikh razak",
  openGraph: {
    title: "Face Morphing Detection System",
    description:
      "Real-time detection of morphed face images using clustering-based ML pipeline.",
    url: "https://face-morphing-detection.vercel.app",
    siteName: "Face Morphing Detection",
    locale: "en_US",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "Face Morphing Detection System",
    description:
      "Real-time detection of morphed face images using clustering-based ML pipeline.",
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      "max-image-preview": "large",
    },
  },
};

export default function RootLayout({ children }) {
  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={`${geistSans.variable} ${geistMono.variable}`}
    >
      <body className="antialiased min-h-screen bg-background text-foreground">
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
