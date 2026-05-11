import { SignIn } from '@clerk/nextjs'
import { ScanFace } from 'lucide-react'

const appearance = {
  variables: {
    colorPrimary: '#7c3aed',
    borderRadius: '0.75rem',
  },
  elements: {
    card: 'shadow-2xl',
    formButtonPrimary: 'bg-gradient-to-r from-purple-600 to-blue-600',
  },
}

export const metadata = {
  title: 'Sign In',
}

export default function SignInPage() {
  return (
    <main className="min-h-screen flex flex-col items-center justify-center bg-gradient-to-br from-background via-background to-purple-950/10 px-4">
      <div className="mb-8 flex flex-col items-center gap-3">
        <div className="rounded-2xl bg-gradient-to-br from-purple-600 to-blue-600 p-3 shadow-lg">
          <ScanFace className="size-8 text-white" aria-hidden />
        </div>
        <div className="text-center">
          <h1 className="text-2xl font-bold tracking-tight">Welcome back to FaceGuard</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Sign in to access your detection dashboard
          </p>
        </div>
      </div>
      <SignIn appearance={appearance} />
    </main>
  )
}
